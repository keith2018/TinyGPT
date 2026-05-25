/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "Scheduler.h"

#include <algorithm>
#include <cinttypes>

#include "ForwardContext.h"
#include "PagedKVCache.h"
#include "Tensor/CachedAllocator.h"
#include "Utils/CUDAUtils.h"
#include "Utils/Logger.h"
#include "kernel/AttentionOps.h"

namespace tt = tinytorch;

namespace tinygpt {

Scheduler::Scheduler(GPTModel& model, PagedKVCache& cache, tokenizer::Tokenizer& tokenizer,
                     const std::vector<int32_t>& baseEosTokenIds, int32_t maxBatchTokens, int32_t prefillChunkSize,
                     int32_t maxGraphBatch)
    : model_(model),
      cache_(cache),
      tokenizer_(tokenizer),
      baseEosTokenIds_(baseEosTokenIds),
      maxBatchTokens_(maxBatchTokens),
      prefillChunkSize_(prefillChunkSize),
      device_(model.device()),
      maxGraphBatch_(maxGraphBatch) {
  ASSERT(device_.isCuda());

  constexpr int32_t kMinPartSize = kernel::kSplitKvMinPartitionSize;
  constexpr int32_t kMaxPartitions = kernel::kSplitKvMaxPartitions;
  const int32_t maxSeqLenKV = cache_.maxBlocksPerSeq() * cache_.blockSize();
  const int32_t numPartitions = std::min((maxSeqLenKV + kMinPartSize - 1) / kMinPartSize, kMaxPartitions);

  auto floatOpts = tt::Options(device_, tt::DType::Float32).noGrad();
  const int64_t splitOSize =
      static_cast<int64_t>(maxBatchTokens_) * model_.numHeads() * numPartitions * model_.headDim();
  const int64_t splitLseSize = static_cast<int64_t>(maxBatchTokens_) * model_.numHeads() * numPartitions * 2;
  splitKvO_ = tt::Tensor({splitOSize}, floatOpts);
  splitKvLse_ = tt::Tensor({splitLseSize}, floatOpts);

  allocateMetaBuffers();
  batchSampler_ = std::make_unique<BatchSampler>(device_, maxBatchTokens_);

  // graph batch sizes: [1, 2, 4] + [8*i for i in range(1, max//8+1)]
  graphBatchSizes_ = {1, 2, 4, 8};
  for (int32_t bs = 16; bs <= maxGraphBatch_; bs += 8) {
    graphBatchSizes_.push_back(bs);
  }

  // padding KV block
  {
    auto padSeqId = cache_.allocate();
    std::vector<int32_t> padSlots;
    cache_.appendTokens(padSeqId, 1, padSlots);
    padKvBlockId_ = padSlots[0] / cache_.blockSize();
  }

  LOGI(
      "Scheduler: maxBatchTokens=%d  prefillChunkSize=%d  maxGraphBatch=%d  "
      "maxSeqLenKV=%d  numPartitions=%d  splitKvO=%.2f MiB  splitKvLse=%.2f MiB  padKvBlock=%d",
      maxBatchTokens_, prefillChunkSize_, maxGraphBatch_, maxSeqLenKV, numPartitions,
      static_cast<double>(splitOSize * 4) / (1024.0 * 1024.0),
      static_cast<double>(splitLseSize * 4) / (1024.0 * 1024.0), padKvBlockId_);

  captureAllGraphs();
}

Scheduler::~Scheduler() { stop(); }

void Scheduler::allocateMetaBuffers() {
  const int64_t maxTok = maxBatchTokens_;
  const int64_t blocksPerSeq = cache_.maxBlocksPerSeq();

  // int64 group:  tokens + positions + lastIdx  = maxTok + maxTok + maxTok
  // int32 group:  slot + cuQ + cuKv + blockTable = maxTok + (maxTok+1) + (maxTok+1) + maxTok*blocksPerSeq
  const int64_t i64Total = 3 * maxTok;
  const int64_t i32Total = maxTok + (maxTok + 1) + (maxTok + 1) + maxTok * blocksPerSeq;

  auto pinnedI32 = tt::Options(tt::Device::cpu(), tt::DType::Int32).noGrad().pinnedMemory(true);
  auto pinnedI64 = tt::Options(tt::Device::cpu(), tt::DType::Int64).noGrad().pinnedMemory(true);
  auto devI32 = tt::Options(device_, tt::DType::Int32).noGrad();
  auto devI64 = tt::Options(device_, tt::DType::Int64).noGrad();

  metaHostI64_ = tt::Tensor({i64Total}, pinnedI64);
  metaDevI64_ = tt::Tensor({i64Total}, devI64);
  metaHostI32_ = tt::Tensor({i32Total}, pinnedI32);
  metaDevI32_ = tt::Tensor({i32Total}, devI32);
}

void Scheduler::captureAllGraphs() {
  tt::NoGradGuard noGrad;

  LOGI("Scheduler: pre-capturing CUDA Graphs for %zu batch sizes [%d..%d]...", graphBatchSizes_.size(),
       graphBatchSizes_.front(), graphBatchSizes_.back());

  const int32_t pageSize = cache_.blockSize();
  const int32_t blocksPerSeq = cache_.maxBlocksPerSeq();
  const int32_t captureMaxSeqLenKV = blocksPerSeq * pageSize;
  auto& stream = tt::cuda::getCurrentCUDAStream(device_.index);

  for (auto it = graphBatchSizes_.rbegin(); it != graphBatchSizes_.rend(); ++it) {
    const int32_t bs = *it;

    auto runner = std::make_unique<CUDAGraphRunner>(model_, device_, bs);

    // dummy metadata: every sequence is a 1-token decode at pos=0, kvLen=1
    const int32_t layoutTokens = bs;
    const int32_t layoutBatch = bs;

    const int64_t offTokens = 0;
    const int64_t offPositions = layoutTokens;
    const int64_t offLastIdx = static_cast<int64_t>(layoutTokens) * 2;

    const int64_t offSlot = 0;
    const int64_t offCuQ = layoutTokens;
    const int64_t offCuKv = offCuQ + layoutBatch + 1;
    const int64_t offBlockTable = offCuKv + layoutBatch + 1;
    const int64_t i64Used = offLastIdx + layoutBatch;
    const int64_t i32Used = offBlockTable + static_cast<int64_t>(layoutBatch) * blocksPerSeq;

    auto* i64Host = metaHostI64_.dataPtr<int64_t>();
    auto* i32Host = metaHostI32_.dataPtr<int32_t>();

    i32Host[offCuQ] = 0;
    i32Host[offCuKv] = 0;
    for (int32_t si = 0; si < bs; si++) {
      i64Host[offTokens + si] = 0;
      i64Host[offPositions + si] = 0;
      i64Host[offLastIdx + si] = si;

      i32Host[offSlot + si] = padKvBlockId_ * pageSize;
      i32Host[offCuQ + si + 1] = i32Host[offCuQ + si] + 1;
      i32Host[offCuKv + si + 1] = i32Host[offCuKv + si] + 1;

      int32_t* btRow = i32Host + offBlockTable + static_cast<size_t>(si) * static_cast<size_t>(blocksPerSeq);
      btRow[0] = padKvBlockId_;
      std::fill(btRow + 1, btRow + blocksPerSeq, -1);
    }

    tt::Storage::copyOnDevice(metaDevI64_.dataPtr<>(), device_, metaHostI64_.dataPtr<>(), tt::Device::cpu(),
                              i64Used * static_cast<int64_t>(sizeof(int64_t)), &stream);
    tt::Storage::copyOnDevice(metaDevI32_.dataPtr<>(), device_, metaHostI32_.dataPtr<>(), tt::Device::cpu(),
                              i32Used * static_cast<int64_t>(sizeof(int32_t)), &stream);
    auto tokensTensor = metaDevI64_.narrow(0, offTokens, layoutTokens);
    auto positionsDev = metaDevI64_.narrow(0, offPositions, layoutTokens);
    auto lastIdxTensor = metaDevI64_.narrow(0, offLastIdx, layoutBatch);
    auto slotDev = metaDevI32_.narrow(0, offSlot, layoutTokens);
    auto cuQDev = metaDevI32_.narrow(0, offCuQ, layoutBatch + 1);
    auto cuKvDev = metaDevI32_.narrow(0, offCuKv, layoutBatch + 1);
    auto blockTableDev = metaDevI32_.narrow(0, offBlockTable, static_cast<int64_t>(layoutBatch) * blocksPerSeq)
                             .view({layoutBatch, blocksPerSeq});

    ForwardContext ctx;
    ctx.pagedCache = &cache_;
    ctx.positions = positionsDev;
    ctx.slotMapping = slotDev;
    ctx.cuSeqLensQ = cuQDev;
    ctx.cuSeqLensKV = cuKvDev;
    ctx.blockTable = blockTableDev;
    ctx.maxSeqLenQ = 1;
    ctx.maxSeqLenKV = captureMaxSeqLenKV;
    ctx.pageSize = pageSize;
    ctx.maxBlocksPerSeq = blocksPerSeq;
    ctx.tmpO = splitKvO_.dataPtr<float>();
    ctx.tmpLse = splitKvLse_.dataPtr<float>();
    ctx.lastTokenIndices = lastIdxTensor;

    // greedy argmax
    auto greedyStage = batchSampler_->makeGreedyStage(bs);
    auto postReplay = [this](tt::cuda::CUDAStream& s) { batchSampler_->recordTokensReady(s); };
    runner->capture(ctx, tokensTensor, greedyStage, stream, std::move(postReplay));

    graphRunners_[bs] = std::move(runner);
  }

  tt::cuda::getCurrentCUDAStream(device_.index).synchronize();
  LOGI("Scheduler: all %zu CUDA Graphs captured successfully", graphBatchSizes_.size());
}

void Scheduler::start() {
  if (running_.exchange(true)) {
    return;
  }
  worker_ = std::thread(&Scheduler::workerLoop, this);
}

void Scheduler::stop() {
  if (!running_.exchange(false)) {
    return;
  }
  waitCV_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
  std::lock_guard<std::mutex> lock(waitMutex_);
  while (!waiting_.empty()) {
    auto& s = waiting_.front();
    s->finishReason = FinishReason::Aborted;
    GenResult r;
    r.promptIds = s->promptIds;
    r.finishReason = FinishReason::Aborted;
    s->promise->set_value(std::move(r));
    waiting_.pop_front();
  }
}

std::shared_future<GenResult> Scheduler::submit(GenRequest req) {
  auto session = std::make_shared<GenSession>(req.samplerConfig);
  session->promptIds = tokenizer_.encode(req.prompt);
  ASSERT(!session->promptIds.empty() && "empty prompt");
  session->stopTokenIds = std::move(req.stopTokenIds);
  session->maxNewTokens = req.maxNewTokens;
  session->onToken = std::move(req.onToken);
  session->promise = std::make_shared<std::promise<GenResult>>();
  auto fut = session->promise->get_future().share();

  {
    std::lock_guard<std::mutex> lock(waitMutex_);
    waiting_.push_back(std::move(session));
  }
  waitCV_.notify_one();
  return fut;
}

size_t Scheduler::numWaiting() const {
  std::lock_guard<std::mutex> lock(waitMutex_);
  return waiting_.size();
}

bool Scheduler::isStopToken(const GenSession& session, int32_t tokenId) const {
  return std::any_of(baseEosTokenIds_.begin(), baseEosTokenIds_.end(), [tokenId](auto id) { return tokenId == id; }) ||
         std::any_of(session.stopTokenIds.begin(), session.stopTokenIds.end(),
                     [tokenId](auto id) { return tokenId == id; });
}

void Scheduler::admitWaiting(std::vector<std::shared_ptr<GenSession>>& active) {
  std::lock_guard<std::mutex> lock(waitMutex_);

  while (!waiting_.empty()) {
    auto& session = waiting_.front();
    const auto promptLen = static_cast<int32_t>(session->promptIds.size());
    const int32_t blocksNeeded = (promptLen + cache_.blockSize() - 1) / cache_.blockSize();
    if (cache_.numFreeBlocks() < blocksNeeded) break;

    // estimate budget using chunk size, not full prompt length
    int32_t usedTokens = 0;
    for (auto& s : active) {
      const auto sPromptLen = static_cast<int32_t>(s->promptIds.size());
      const int32_t sRemaining = sPromptLen - s->numPromptForwarded;
      usedTokens += (sRemaining > 0) ? std::min(sRemaining, prefillChunkSize_) : 1;
    }
    const int32_t estimatedCost = std::min(promptLen, prefillChunkSize_);
    if (usedTokens + estimatedCost > maxBatchTokens_) break;

    session->kvSeqId = cache_.allocate();
    active.push_back(std::move(session));
    waiting_.pop_front();
  }
}

void Scheduler::completeSession(std::shared_ptr<GenSession>& session) {
  if (session->onToken && !session->aborted) {
    std::string rest = tokenizer::Tokenizer::decodeStreamFlush(session->streamState);
    if (!rest.empty()) {
      bool isOk = session->onToken(rest);
      UNUSED(isOk);
    }
  }

  GenResult result;
  result.promptIds = std::move(session->promptIds);
  result.generatedIds = std::move(session->generatedIds);
  result.finishReason = session->finishReason;
  result.text = tokenizer_.decode(result.generatedIds);
  session->promise->set_value(std::move(result));

  cache_.free(session->kvSeqId);
  session->finished = true;
}

void Scheduler::retireFinished(std::vector<std::shared_ptr<GenSession>>& active) {
  for (auto it = active.begin(); it != active.end();) {
    if ((*it)->finished) {
      completeSession(*it);
      it = active.erase(it);
    } else {
      ++it;
    }
  }
}

void Scheduler::runStep(std::vector<std::shared_ptr<GenSession>>& active) {
  tt::NoGradGuard noGrad;

  // harvest previous step's token IDs (phase 1: critical path),
  // then process callbacks (phase 2: deferrable) after launching this step's GPU work.
  harvestTokenIds();

  retireFinished(active);
  if (active.empty()) {
    processCallbacks();  // flush any pending callbacks before sleeping
    return;
  }

  const int32_t pageSize = cache_.blockSize();
  const int32_t blocksPerSeq = cache_.maxBlocksPerSeq();
  const auto btStride = static_cast<size_t>(blocksPerSeq);

  // chunked_prefill two-phase budget: decode gets 1 token each, prefill fills remainder
  int32_t budget = maxBatchTokens_;

  // pass 1: classify sessions and reserve decode budget
  struct ScheduleEntry {
    int32_t activeIdx;
    int32_t qLen;
  };
  std::vector<ScheduleEntry> decodeEntries;
  std::vector<int32_t> prefillActiveIndices;
  decodeEntries.reserve(active.size());
  prefillActiveIndices.reserve(active.size());

  for (int32_t i = 0; i < static_cast<int32_t>(active.size()); i++) {
    GenSession& s = *active[static_cast<size_t>(i)];
    const auto promptLen = static_cast<int32_t>(s.promptIds.size());
    const int32_t remainingPrompt = promptLen - s.numPromptForwarded;
    if (remainingPrompt <= 0) {
      decodeEntries.push_back({i, 1});
      budget -= 1;
    } else {
      prefillActiveIndices.push_back(i);
    }
  }

  // pass 2: distribute remaining budget to prefill chunks
  std::vector<ScheduleEntry> prefillEntries;
  prefillEntries.reserve(prefillActiveIndices.size());
  for (int32_t idx : prefillActiveIndices) {
    if (budget <= 0) break;
    GenSession& s = *active[static_cast<size_t>(idx)];
    const auto promptLen = static_cast<int32_t>(s.promptIds.size());
    const int32_t remaining = promptLen - s.numPromptForwarded;
    const int32_t chunk = std::min({remaining, prefillChunkSize_, budget});
    if (chunk <= 0) break;
    prefillEntries.push_back({idx, chunk});
    budget -= chunk;
  }

  // merge: decode first, then prefill
  std::vector<ScheduleEntry> allEntries;
  allEntries.reserve(decodeEntries.size() + prefillEntries.size());
  allEntries.insert(allEntries.end(), decodeEntries.begin(), decodeEntries.end());
  allEntries.insert(allEntries.end(), prefillEntries.begin(), prefillEntries.end());

  if (allEntries.empty()) {
    processCallbacks();
    return;
  }

  // allocate KV cache slots and compute metadata
  std::vector<int32_t> scheduled;
  std::vector<std::vector<int32_t>> slotsPerSeq;
  std::vector<int32_t> qLens;
  scheduled.reserve(allEntries.size());
  slotsPerSeq.reserve(allEntries.size());
  qLens.reserve(allEntries.size());

  int32_t totalTokens = 0;
  int32_t maxSeqLenQ = 0;
  int32_t maxSeqLenKV = 0;

  for (auto& entry : allEntries) {
    GenSession& s = *active[static_cast<size_t>(entry.activeIdx)];
    const int32_t qLen = entry.qLen;

    std::vector<int32_t> slots;
    if (!cache_.appendTokens(s.kvSeqId, qLen, slots)) {
      LOGW("Scheduler: PagedKVCache exhausted for session — ending with Length (free=%" PRId64 ")",
           cache_.numFreeBlocks());
      s.finishReason = FinishReason::Length;
      s.finished = true;
      continue;
    }

    ASSERT(totalTokens + qLen <= maxBatchTokens_);
    const int32_t kvLen = s.numCachedTokens + qLen;
    maxSeqLenQ = std::max(maxSeqLenQ, qLen);
    maxSeqLenKV = std::max(maxSeqLenKV, kvLen);
    totalTokens += qLen;
    scheduled.push_back(entry.activeIdx);
    qLens.push_back(qLen);
    slotsPerSeq.push_back(std::move(slots));
  }

  const auto scheduledBatch = static_cast<int32_t>(scheduled.size());
  if (scheduledBatch == 0) {
    return;
  }

  // collect sampler pointers (in scheduled order) for the batch sampler
  std::vector<Sampler*> batchSamplers(static_cast<size_t>(scheduledBatch));
  for (int32_t si = 0; si < scheduledBatch; si++) {
    const int32_t ai = scheduled[static_cast<size_t>(si)];
    batchSamplers[static_cast<size_t>(si)] = &active[static_cast<size_t>(ai)]->sampler;
  }
  const bool allGreedy = BatchSampler::allGreedy(batchSamplers);

  // determine if CUDA graph path will be used (affects buffer layout).
  const bool allDecode = (totalTokens == scheduledBatch);
  int32_t graphBatch = 0;
  if (allDecode && scheduledBatch <= maxGraphBatch_ && padKvBlockId_ >= 0) {
    for (int32_t bs : graphBatchSizes_) {
      if (bs >= scheduledBatch) {
        graphBatch = bs;
        break;
      }
    }
  }

  // for graph path, use padded sizes so captured graph sees same buffer addresses
  const int32_t layoutTokens = (graphBatch > 0) ? graphBatch : totalTokens;
  const int32_t layoutBatch = (graphBatch > 0) ? graphBatch : scheduledBatch;

  // int64: tokens(T) | positions(T) | lastIdx(B)
  // int32: slot(T)   | cuQ(B+1)     | cuKv(B+1) | blockTable(B * blocksPerSeq)
  const int64_t offTokens = 0;
  const int64_t offPositions = layoutTokens;
  const int64_t offLastIdx = static_cast<int64_t>(layoutTokens) * 2;
  const int64_t i64Used = offLastIdx + layoutBatch;

  const int64_t offSlot = 0;
  const int64_t offCuQ = layoutTokens;
  const int64_t offCuKv = offCuQ + layoutBatch + 1;
  const int64_t offBlockTable = offCuKv + layoutBatch + 1;
  const int64_t i32Used = offBlockTable + static_cast<int64_t>(layoutBatch) * blocksPerSeq;

  auto* i64Host = metaHostI64_.dataPtr<int64_t>();
  auto* i32Host = metaHostI32_.dataPtr<int32_t>();
  auto* tokensH = i64Host + offTokens;
  auto* posH = i64Host + offPositions;
  auto* lastIdxH = i64Host + offLastIdx;
  auto* slotH = i32Host + offSlot;
  auto* cuQH = i32Host + offCuQ;
  auto* cuKvH = i32Host + offCuKv;
  auto* btH = i32Host + offBlockTable;

  // pack all fields into their dense slots
  cuQH[0] = 0;
  cuKvH[0] = 0;
  int32_t writeTok = 0;
  for (int32_t si = 0; si < scheduledBatch; si++) {
    const int32_t ai = scheduled[static_cast<size_t>(si)];
    GenSession& s = *active[static_cast<size_t>(ai)];
    const int32_t qLen = qLens[static_cast<size_t>(si)];
    const auto promptLen = static_cast<int32_t>(s.promptIds.size());
    const int32_t remainingPrompt = promptLen - s.numPromptForwarded;
    const auto& slots = slotsPerSeq[static_cast<size_t>(si)];

    for (int32_t t = 0; t < qLen; t++) {
      const int32_t pos = writeTok + t;
      tokensH[pos] = (remainingPrompt > 0) ? s.promptIds[s.numPromptForwarded + t] : s.lastToken;
      posH[pos] = s.numCachedTokens + t;
      slotH[pos] = slots[static_cast<size_t>(t)];
    }
    writeTok += qLen;

    int32_t* btRow = btH + static_cast<size_t>(si) * btStride;
    const auto& blocks = cache_.blocksOf(s.kvSeqId);
    const size_t copyCnt = std::min(blocks.size(), btStride);
    std::copy_n(blocks.begin(), copyCnt, btRow);
    std::fill(btRow + copyCnt, btRow + btStride, -1);

    const int32_t kvLen = s.numCachedTokens + qLen;
    cuQH[si + 1] = cuQH[si] + qLen;
    cuKvH[si + 1] = cuKvH[si] + kvLen;
    lastIdxH[si] = cuQH[si + 1] - 1;  // index of this seq's last logit row

    if (remainingPrompt > 0) s.numPromptForwarded += qLen;
    s.numCachedTokens += qLen;
  }

  // if graph path, fill padding before H2D
  if (graphBatch > 0 && scheduledBatch < graphBatch) {
    padMetadataForGraph(scheduledBatch, graphBatch, totalTokens);
  }

  auto& stream = tt::cuda::getCurrentCUDAStream(device_.index);
  tt::Storage::copyOnDevice(metaDevI64_.dataPtr<>(), device_, metaHostI64_.dataPtr<>(), tt::Device::cpu(),
                            i64Used * static_cast<int64_t>(sizeof(int64_t)), &stream);
  tt::Storage::copyOnDevice(metaDevI32_.dataPtr<>(), device_, metaHostI32_.dataPtr<>(), tt::Device::cpu(),
                            i32Used * static_cast<int64_t>(sizeof(int32_t)), &stream);

  // tensor views
  auto tokensTensor = metaDevI64_.narrow(0, offTokens, layoutTokens);
  auto positionsDev = metaDevI64_.narrow(0, offPositions, layoutTokens);
  auto lastIdxTensor = metaDevI64_.narrow(0, offLastIdx, layoutBatch);
  auto slotDev = metaDevI32_.narrow(0, offSlot, layoutTokens);
  auto cuQDev = metaDevI32_.narrow(0, offCuQ, layoutBatch + 1);
  auto cuKvDev = metaDevI32_.narrow(0, offCuKv, layoutBatch + 1);
  auto blockTableDev = metaDevI32_.narrow(0, offBlockTable, static_cast<int64_t>(layoutBatch) * blocksPerSeq)
                           .view({layoutBatch, blocksPerSeq});

  ForwardContext ctx;
  ctx.pagedCache = &cache_;
  ctx.positions = positionsDev;
  ctx.slotMapping = slotDev;
  ctx.cuSeqLensQ = cuQDev;
  ctx.cuSeqLensKV = cuKvDev;
  ctx.blockTable = blockTableDev;
  ctx.maxSeqLenQ = maxSeqLenQ;
  ctx.maxSeqLenKV = maxSeqLenKV;
  ctx.pageSize = pageSize;
  ctx.maxBlocksPerSeq = blocksPerSeq;
  ctx.tmpO = splitKvO_.dataPtr<float>();
  ctx.tmpLse = splitKvLse_.dataPtr<float>();
  ctx.lastTokenIndices = lastIdxTensor;

  // dispatch: graph replay (greedy decode only) or eager forward + sample
  bool dispatched = false;
  if (graphBatch > 0 && allGreedy) {
    auto it = graphRunners_.find(graphBatch);
    if (it != graphRunners_.end() && it->second && it->second->captured()) {
      it->second->replay(stream);  // postReplay hook records sampler ready event
      dispatched = true;
    }
  }

  if (!dispatched) {
    // eager path (prefill, mixed, or non-greedy decode)
    auto eagerTokens = (graphBatch > 0) ? metaDevI64_.narrow(0, offTokens, totalTokens) : tokensTensor;
    auto eagerLastIdx = (graphBatch > 0) ? metaDevI64_.narrow(0, offLastIdx, scheduledBatch) : lastIdxTensor;

    ForwardContext eagerCtx = ctx;
    if (graphBatch > 0) {
      eagerCtx.positions = metaDevI64_.narrow(0, offPositions, totalTokens);
      eagerCtx.slotMapping = metaDevI32_.narrow(0, offSlot, totalTokens);
      eagerCtx.cuSeqLensQ = metaDevI32_.narrow(0, offCuQ, scheduledBatch + 1);
      eagerCtx.cuSeqLensKV = metaDevI32_.narrow(0, offCuKv, scheduledBatch + 1);
      eagerCtx.blockTable = metaDevI32_.narrow(0, offBlockTable, static_cast<int64_t>(scheduledBatch) * blocksPerSeq)
                                .view({scheduledBatch, blocksPerSeq});
      eagerCtx.lastTokenIndices = eagerLastIdx;
    }

    tt::Tensor logits;
    {
      ForwardContextGuard guard(&eagerCtx);
      logits = model_.forward(eagerTokens);  // [scheduledBatch, vocabSize]
    }
    batchSampler_->sampleEager(logits, batchSamplers, allGreedy, stream);
  }

  prev_.sessions.clear();
  prev_.sessions.reserve(static_cast<size_t>(scheduledBatch));
  for (int32_t si = 0; si < scheduledBatch; si++) {
    const int32_t ai = scheduled[static_cast<size_t>(si)];
    prev_.sessions.push_back(active[static_cast<size_t>(ai)]);
  }
  prev_.scheduledBatch = scheduledBatch;
  prev_.hasPendingHarvest = true;

  // process streaming callbacks after launching GPU work (overlaps with forward)
  processCallbacks();

  retireFinished(active);
}

void Scheduler::harvestTokenIds() {
  if (!prev_.hasPendingHarvest) {
    return;
  }

  const int64_t* sampledH = batchSampler_->consumeTokens();
  pendingCallbacks_.clear();

  for (int32_t si = 0; si < prev_.scheduledBatch; si++) {
    GenSession& s = *prev_.sessions[static_cast<size_t>(si)];
    if (s.finished) {
      continue;
    }

    const auto tokenId = static_cast<int32_t>(sampledH[si]);
    s.lastToken = tokenId;

    if (s.numPromptForwarded != static_cast<int32_t>(s.promptIds.size())) {
      continue;
    }

    if (isStopToken(s, tokenId)) {
      s.finishReason = FinishReason::Stop;
      s.finished = true;
      continue;
    }

    s.generatedIds.push_back(tokenId);

    // defer callback to processCallbacks() — runs while GPU is busy
    if (s.onToken) {
      pendingCallbacks_.push_back({prev_.sessions[static_cast<size_t>(si)], tokenId});
    }

    if (!s.finished && static_cast<int32_t>(s.generatedIds.size()) >= s.maxNewTokens) {
      s.finishReason = FinishReason::Length;
      s.finished = true;
    }
  }

  prev_.sessions.clear();
  prev_.scheduledBatch = 0;
  prev_.hasPendingHarvest = false;
}

void Scheduler::processCallbacks() {
  for (auto& [session, tokenId] : pendingCallbacks_) {
    GenSession& s = *session;
    if (s.finished) continue;

    std::vector<int32_t> one{tokenId};
    std::string chunk = tokenizer_.decodeStream(one, s.streamState);
    if (!chunk.empty() && !s.onToken(chunk)) {
      s.aborted = true;
      s.finishReason = FinishReason::Stop;
      s.finished = true;
    }
  }
  pendingCallbacks_.clear();
}

void Scheduler::padMetadataForGraph(int32_t realBatch, int32_t graphBatch, int32_t realTokens) {
  if (realBatch >= graphBatch) return;

  const int32_t pageSize = cache_.blockSize();
  const int32_t blocksPerSeq = cache_.maxBlocksPerSeq();
  const auto btStride = static_cast<size_t>(blocksPerSeq);
  const int32_t padCount = graphBatch - realBatch;
  const int32_t graphTokens = graphBatch;

  const int64_t offTokens = 0;
  const int64_t offPositions = graphTokens;
  const int64_t offLastIdx = static_cast<int64_t>(graphTokens) * 2;

  const int64_t offSlot = 0;
  const int64_t offCuQ = graphTokens;
  const int64_t offCuKv = offCuQ + graphBatch + 1;
  const int64_t offBlockTable = offCuKv + graphBatch + 1;

  auto* i64Host = metaHostI64_.dataPtr<int64_t>();
  auto* i32Host = metaHostI32_.dataPtr<int32_t>();

  // fill padding entries [realBatch, graphBatch)
  for (int32_t pi = 0; pi < padCount; pi++) {
    const int32_t si = realBatch + pi;
    const int32_t ti = realTokens + pi;

    i64Host[offTokens + ti] = 0;
    i64Host[offPositions + ti] = 0;
    i64Host[offLastIdx + si] = ti;

    i32Host[offSlot + ti] = padKvBlockId_ * pageSize;
    i32Host[offCuQ + si + 1] = i32Host[offCuQ + si] + 1;
    i32Host[offCuKv + si + 1] = i32Host[offCuKv + si] + 1;

    int32_t* btRow = i32Host + offBlockTable + static_cast<size_t>(si) * btStride;
    btRow[0] = padKvBlockId_;
    std::fill(btRow + 1, btRow + btStride, -1);
  }
}

void Scheduler::workerLoop() {
  std::vector<std::shared_ptr<GenSession>> active;

  while (running_.load()) {
    admitWaiting(active);

    if (active.empty()) {
      runningCount_.store(0);
      std::unique_lock<std::mutex> lock(waitMutex_);
      waitCV_.wait(lock, [&] { return !waiting_.empty() || !running_.load(); });
      continue;
    }

    runningCount_.store(active.size());
    runStep(active);
  }

  // drain pending harvest before shutdown
  harvestTokenIds();
  processCallbacks();
  retireFinished(active);

  runningCount_.store(0);
  for (auto& s : active) {
    s->finishReason = FinishReason::Aborted;
    completeSession(s);
  }
}

}  // namespace tinygpt
