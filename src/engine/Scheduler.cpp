/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "Scheduler.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>

#include "ForwardContext.h"
#include "Functions.h"
#include "PagedKVCache.h"
#include "Tensor/CachedAllocator.h"
#include "Utils/CUDAUtils.h"
#include "Utils/Logger.h"
#include "Utils/RandomGenerator.h"
#include "kernel/AttentionOps.h"
#include "kernel/SamplerOps.h"

namespace tt = tinytorch;

namespace tinygpt {

struct Scheduler::CudaPipeState {
  tinytorch::cuda::CUDAEvent sampledReady;
  bool eventInit = false;
};

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
      cudaPipe_(std::make_unique<CudaPipeState>()),
      maxGraphBatch_(maxGraphBatch) {
  ASSERT(device_.isCuda());

  constexpr int32_t kMinPartSize = kernel::kSplitKvMinPartitionSize;
  constexpr int32_t kMaxPartitions = kernel::kSplitKvMaxPartitions;
  const int32_t maxSeqLenKV = cache_.maxBlocksPerSeq() * cache_.blockSize();
  const int32_t numPartitions = std::min((maxSeqLenKV + kMinPartSize - 1) / kMinPartSize, kMaxPartitions);

  auto floatOpts = tt::Options(device_, tt::DType::Float32).noGrad();
  const int64_t tmpOSize = static_cast<int64_t>(maxBatchTokens_) * model_.numHeads() * numPartitions * model_.headDim();
  const int64_t tmpLseSize = static_cast<int64_t>(maxBatchTokens_) * model_.numHeads() * numPartitions * 2;
  tmpO_ = tt::Tensor({tmpOSize}, floatOpts);
  tmpLse_ = tt::Tensor({tmpLseSize}, floatOpts);

  allocateMetaBuffers();

  // graph batch sizes: [1, 2, 4] + [8*i for i in range(1, max//8+1)]
  graphBatchSizes_ = {1, 2, 4, 8};
  for (int32_t bs = 16; bs <= maxGraphBatch_; bs += 8) {
    graphBatchSizes_.push_back(bs);
  }

  // allocate a dummy KV block for graph padding
  {
    auto dummySeqId = cache_.allocate();
    std::vector<int32_t> dummySlots;
    cache_.appendTokens(dummySeqId, 1, dummySlots);
    dummyBlockId_ = dummySlots[0] / cache_.blockSize();
  }

  LOGI(
      "Scheduler: maxBatchTokens=%d  prefillChunkSize=%d  maxGraphBatch=%d  "
      "maxSeqLenKV=%d  numPartitions=%d  tmpO=%.2f MiB  tmpLse=%.2f MiB  dummyBlock=%d",
      maxBatchTokens_, prefillChunkSize_, maxGraphBatch_, maxSeqLenKV, numPartitions,
      static_cast<double>(tmpOSize * 4) / (1024.0 * 1024.0), static_cast<double>(tmpLseSize * 4) / (1024.0 * 1024.0),
      dummyBlockId_);

  // pre-capture all graphs at startup
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

  hostI64_ = tt::Tensor({i64Total}, pinnedI64);
  devI64_ = tt::Tensor({i64Total}, devI64);
  hostI32_ = tt::Tensor({i32Total}, pinnedI32);
  devI32_ = tt::Tensor({i32Total}, devI32);
  sampledHost_ = tt::Tensor({maxTok}, pinnedI64);
}

// pre-capture all CUDA graphs (largest → smallest so first capture sizes the pool)
void Scheduler::captureAllGraphs() {
  tt::NoGradGuard noGrad;

  LOGI("Scheduler: pre-capturing CUDA Graphs for %zu batch sizes [%d..%d]...", graphBatchSizes_.size(),
       graphBatchSizes_.front(), graphBatchSizes_.back());

  const int32_t pageSize = cache_.blockSize();
  const int32_t blocksPerSeq = cache_.maxBlocksPerSeq();
  const int32_t captureMaxSeqLenKV = blocksPerSeq * pageSize;

  if (!cudaPipe_->eventInit) {
    cudaPipe_->sampledReady = tt::cuda::createCUDAEvent(device_.index);
    cudaPipe_->eventInit = true;
  }

  auto& stream = tt::cuda::getCurrentCUDAStream(device_.index);

  // iterate largest → smallest
  for (auto it = graphBatchSizes_.rbegin(); it != graphBatchSizes_.rend(); ++it) {
    const int32_t bs = *it;

    auto runner = std::make_unique<CUDAGraphRunner>(model_, device_, bs);

    // fill dummy metadata: all sequences are decode with token=0, pos=0, kvLen=1
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

    auto* i64Host = hostI64_.dataPtr<int64_t>();
    auto* i32Host = hostI32_.dataPtr<int32_t>();

    // fill dummy data
    i32Host[offCuQ] = 0;
    i32Host[offCuKv] = 0;
    for (int32_t si = 0; si < bs; si++) {
      i64Host[offTokens + si] = 0;
      i64Host[offPositions + si] = 0;
      i64Host[offLastIdx + si] = si;

      i32Host[offSlot + si] = dummyBlockId_ * pageSize;
      i32Host[offCuQ + si + 1] = i32Host[offCuQ + si] + 1;
      i32Host[offCuKv + si + 1] = i32Host[offCuKv + si] + 1;

      // blockTable row
      int32_t* btRow = i32Host + offBlockTable + static_cast<size_t>(si) * static_cast<size_t>(blocksPerSeq);
      btRow[0] = dummyBlockId_;
      std::fill(btRow + 1, btRow + blocksPerSeq, -1);
    }

    tt::Storage::copyOnDevice(devI64_.dataPtr<>(), device_, hostI64_.dataPtr<>(), tt::Device::cpu(),
                              i64Used * static_cast<int64_t>(sizeof(int64_t)), &stream);
    tt::Storage::copyOnDevice(devI32_.dataPtr<>(), device_, hostI32_.dataPtr<>(), tt::Device::cpu(),
                              i32Used * static_cast<int64_t>(sizeof(int32_t)), &stream);
    auto tokensTensor = devI64_.narrow(0, offTokens, layoutTokens);
    auto positionsDev = devI64_.narrow(0, offPositions, layoutTokens);
    auto lastIdxTensor = devI64_.narrow(0, offLastIdx, layoutBatch);
    auto slotDev = devI32_.narrow(0, offSlot, layoutTokens);
    auto cuQDev = devI32_.narrow(0, offCuQ, layoutBatch + 1);
    auto cuKvDev = devI32_.narrow(0, offCuKv, layoutBatch + 1);
    auto blockTableDev = devI32_.narrow(0, offBlockTable, static_cast<int64_t>(layoutBatch) * blocksPerSeq)
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
    ctx.tmpO = tmpO_.dataPtr<float>();
    ctx.tmpLse = tmpLse_.dataPtr<float>();
    ctx.lastTokenIndices = lastIdxTensor;

    // capture (greedy argmax — most common for multi-batch decode)
    runner->capture(ctx, tokensTensor, sampledHost_, static_cast<void*>(&cudaPipe_->sampledReady),
                    static_cast<void*>(&stream), nullptr);

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
  auto seq = std::make_shared<Seq>(req.samplerConfig);
  seq->promptIds = tokenizer_.encode(req.prompt);
  ASSERT(!seq->promptIds.empty() && "empty prompt");
  seq->stopTokenIds = std::move(req.stopTokenIds);
  seq->maxNewTokens = req.maxNewTokens;
  seq->onToken = std::move(req.onToken);
  seq->promise = std::make_shared<std::promise<GenResult>>();
  auto fut = seq->promise->get_future().share();

  {
    std::lock_guard<std::mutex> lock(waitMutex_);
    waiting_.push_back(std::move(seq));
  }
  waitCV_.notify_one();
  return fut;
}

size_t Scheduler::numWaiting() const {
  std::lock_guard<std::mutex> lock(waitMutex_);
  return waiting_.size();
}

bool Scheduler::isEos(const Seq& seq, int32_t tokenId) const {
  for (auto id : baseEosTokenIds_) {
    if (tokenId == id) return true;
  }
  for (auto id : seq.stopTokenIds) {
    if (tokenId == id) return true;
  }
  return false;
}

bool Scheduler::admitWaiting(std::vector<std::shared_ptr<Seq>>& active) {
  std::lock_guard<std::mutex> lock(waitMutex_);
  bool admittedAny = false;

  while (!waiting_.empty()) {
    auto& seq = waiting_.front();
    const auto promptLen = static_cast<int32_t>(seq->promptIds.size());
    const int32_t blocksNeeded = (promptLen + cache_.blockSize() - 1) / cache_.blockSize();
    if (cache_.numFreeBlocks() < blocksNeeded) break;

    // estimate budget using chunk size, not full prompt length
    int32_t usedTokens = 0;
    for (auto& s : active) {
      const auto sPromptLen = static_cast<int32_t>(s->promptIds.size());
      const int32_t sRemaining = sPromptLen - s->numPromptProcessed;
      usedTokens += (sRemaining > 0) ? std::min(sRemaining, prefillChunkSize_) : 1;
    }
    const int32_t estimatedCost = std::min(promptLen, prefillChunkSize_);
    if (usedTokens + estimatedCost > maxBatchTokens_) break;

    seq->cacheId = cache_.allocate();
    active.push_back(std::move(seq));
    waiting_.pop_front();
    admittedAny = true;
  }
  return admittedAny;
}

void Scheduler::completeSeq(std::shared_ptr<Seq>& seq) {
  if (seq->onToken && !seq->aborted) {
    std::string rest = tokenizer::Tokenizer::decodeStreamFlush(seq->streamState);
    if (!rest.empty()) {
      seq->onToken(rest);
    }
  }

  GenResult result;
  result.promptIds = std::move(seq->promptIds);
  result.generatedIds = std::move(seq->generatedIds);
  result.finishReason = seq->finishReason;
  result.text = tokenizer_.decode(result.generatedIds);
  seq->promise->set_value(std::move(result));

  cache_.free(seq->cacheId);
  seq->finished = true;
}

void Scheduler::sweepFinished(std::vector<std::shared_ptr<Seq>>& active) {
  for (auto it = active.begin(); it != active.end();) {
    if ((*it)->finished) {
      completeSeq(*it);
      it = active.erase(it);
    } else {
      ++it;
    }
  }
}

void Scheduler::runStep(std::vector<std::shared_ptr<Seq>>& active) {
  tt::NoGradGuard noGrad;

  // harvest previous step's token IDs (phase 1: critical path),
  // then process callbacks (phase 2: deferrable) after launching this step's GPU work.

  harvestTokenIds();

  sweepFinished(active);
  if (active.empty()) {
    processCallbacks();  // flush any pending callbacks before sleeping
    return;
  }

  const int32_t pageSize = cache_.blockSize();
  const int32_t blocksPerSeq = cache_.maxBlocksPerSeq();
  const auto btStride = static_cast<size_t>(blocksPerSeq);

  // chunked_prefill two-phase budget: decode gets 1 token each, prefill fills remainder
  int32_t budget = maxBatchTokens_;

  // pass 1: classify sequences and reserve decode budget
  struct ScheduleEntry {
    int32_t activeIdx;
    int32_t qLen;
  };
  std::vector<ScheduleEntry> decodeEntries;
  std::vector<int32_t> prefillActiveIndices;
  decodeEntries.reserve(active.size());
  prefillActiveIndices.reserve(active.size());

  for (int32_t i = 0; i < static_cast<int32_t>(active.size()); i++) {
    Seq& s = *active[static_cast<size_t>(i)];
    const auto promptLen = static_cast<int32_t>(s.promptIds.size());
    const int32_t remainingPrompt = promptLen - s.numPromptProcessed;
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
    Seq& s = *active[static_cast<size_t>(idx)];
    const auto promptLen = static_cast<int32_t>(s.promptIds.size());
    const int32_t remaining = promptLen - s.numPromptProcessed;
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
    Seq& s = *active[static_cast<size_t>(entry.activeIdx)];
    const int32_t qLen = entry.qLen;

    std::vector<int32_t> slots;
    if (!cache_.appendTokens(s.cacheId, qLen, slots)) {
      LOGW("Scheduler: PagedKVCache exhausted for seq — ending with Length (free=%" PRId64 ")", cache_.numFreeBlocks());
      s.finishReason = FinishReason::Length;
      s.finished = true;
      continue;
    }

    ASSERT(totalTokens + qLen <= maxBatchTokens_);
    const int32_t kvLen = s.pastLen + qLen;
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

  // determine if CUDA graph path will be used (affects buffer layout)
  const bool allDecode = (totalTokens == scheduledBatch);
  int32_t graphBatch = 0;
  if (allDecode && scheduledBatch <= maxGraphBatch_ && dummyBlockId_ >= 0) {
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

  auto* i64Host = hostI64_.dataPtr<int64_t>();
  auto* i32Host = hostI32_.dataPtr<int32_t>();
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
    Seq& s = *active[static_cast<size_t>(ai)];
    const int32_t qLen = qLens[static_cast<size_t>(si)];
    const auto promptLen = static_cast<int32_t>(s.promptIds.size());
    const int32_t remainingPrompt = promptLen - s.numPromptProcessed;
    const auto& slots = slotsPerSeq[static_cast<size_t>(si)];

    for (int32_t t = 0; t < qLen; t++) {
      const int32_t pos = writeTok + t;
      tokensH[pos] = (remainingPrompt > 0) ? s.promptIds[static_cast<size_t>(s.numPromptProcessed + t)] : s.lastToken;
      posH[pos] = s.pastLen + t;
      slotH[pos] = slots[static_cast<size_t>(t)];
    }
    writeTok += qLen;

    int32_t* btRow = btH + static_cast<size_t>(si) * btStride;
    const auto& blocks = cache_.blocksOf(s.cacheId);
    const size_t copyCnt = std::min(blocks.size(), btStride);
    std::copy_n(blocks.begin(), copyCnt, btRow);
    std::fill(btRow + copyCnt, btRow + btStride, -1);

    const int32_t kvLen = s.pastLen + qLen;
    cuQH[si + 1] = cuQH[si] + qLen;
    cuKvH[si + 1] = cuKvH[si] + kvLen;
    lastIdxH[si] = cuQH[si + 1] - 1;  // index of this seq's last logit row

    if (remainingPrompt > 0) s.numPromptProcessed += qLen;
    s.pastLen += qLen;
  }

  // if graph path, fill padding before H2D
  if (graphBatch > 0 && scheduledBatch < graphBatch) {
    padMetadataForGraph(scheduledBatch, graphBatch, totalTokens);
  }

  auto& stream = tt::cuda::getCurrentCUDAStream(device_.index);
  tt::Storage::copyOnDevice(devI64_.dataPtr<>(), device_, hostI64_.dataPtr<>(), tt::Device::cpu(),
                            i64Used * static_cast<int64_t>(sizeof(int64_t)), &stream);
  tt::Storage::copyOnDevice(devI32_.dataPtr<>(), device_, hostI32_.dataPtr<>(), tt::Device::cpu(),
                            i32Used * static_cast<int64_t>(sizeof(int32_t)), &stream);

  // tensor views
  auto tokensTensor = devI64_.narrow(0, offTokens, layoutTokens);
  auto positionsDev = devI64_.narrow(0, offPositions, layoutTokens);
  auto lastIdxTensor = devI64_.narrow(0, offLastIdx, layoutBatch);
  auto slotDev = devI32_.narrow(0, offSlot, layoutTokens);
  auto cuQDev = devI32_.narrow(0, offCuQ, layoutBatch + 1);
  auto cuKvDev = devI32_.narrow(0, offCuKv, layoutBatch + 1);
  auto blockTableDev = devI32_.narrow(0, offBlockTable, static_cast<int64_t>(layoutBatch) * blocksPerSeq)
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
  ctx.tmpO = tmpO_.dataPtr<float>();
  ctx.tmpLse = tmpLse_.dataPtr<float>();

  // select last-token rows before lm_head to skip full-vocab projection on non-last tokens
  ctx.lastTokenIndices = lastIdxTensor;

  if (!cudaPipe_->eventInit) {
    cudaPipe_->sampledReady = tt::cuda::createCUDAEvent(device_.index);
    cudaPipe_->eventInit = true;
  }

  bool allGreedy = true;
  for (int32_t si = 0; si < scheduledBatch; si++) {
    const int32_t ai = scheduled[static_cast<size_t>(si)];
    if (active[static_cast<size_t>(ai)]->sampler.doSample()) {
      allGreedy = false;
      break;
    }
  }

  // graph replay path (greedy decode only)
  const bool canUseGraph = (graphBatch > 0) && allGreedy;

  if (canUseGraph) {
    auto it = graphRunners_.find(graphBatch);
    if (it != graphRunners_.end() && it->second && it->second->captured()) {
      it->second->replay(static_cast<void*>(&stream));
      goto post_forward;
    }
  }

  {
    // eager path (prefill, mixed, or non-greedy decode)
    auto eagerTokens = (graphBatch > 0) ? devI64_.narrow(0, offTokens, totalTokens) : tokensTensor;
    auto eagerLastIdx = (graphBatch > 0) ? devI64_.narrow(0, offLastIdx, scheduledBatch) : lastIdxTensor;

    ForwardContext eagerCtx = ctx;
    if (graphBatch > 0) {
      eagerCtx.positions = devI64_.narrow(0, offPositions, totalTokens);
      eagerCtx.slotMapping = devI32_.narrow(0, offSlot, totalTokens);
      eagerCtx.cuSeqLensQ = devI32_.narrow(0, offCuQ, scheduledBatch + 1);
      eagerCtx.cuSeqLensKV = devI32_.narrow(0, offCuKv, scheduledBatch + 1);
      eagerCtx.blockTable = devI32_.narrow(0, offBlockTable, static_cast<int64_t>(scheduledBatch) * blocksPerSeq)
                                .view({scheduledBatch, blocksPerSeq});
      eagerCtx.lastTokenIndices = eagerLastIdx;
    }

    tt::Tensor logits;
    {
      ForwardContextGuard guard(&eagerCtx);
      logits = model_.forward(eagerTokens);  // [scheduledBatch, vocabSize]
    }

    tt::Tensor sampledDev;
    if (allGreedy) {
      sampledDev = tt::function::argmax(logits, -1, true);
    } else {
      std::vector<kernel::SamplingParams> perRow(scheduledBatch);
      for (int32_t si = 0; si < scheduledBatch; si++) {
        const int32_t ai = scheduled[static_cast<size_t>(si)];
        perRow[static_cast<size_t>(si)] = active[static_cast<size_t>(ai)]->sampler.params();
      }
      const auto globalSeed = tt::RandomGeneratorCUDA::getSeed();
      const auto globalSeq = tt::RandomGeneratorCUDA::nextSequence();
      sampledDev = kernel::fusedSample(logits, perRow.data(), scheduledBatch, globalSeed, globalSeq);
    }

    tt::Storage::copyOnDevice(sampledHost_.dataPtr<>(), tt::Device::cpu(), sampledDev.dataPtr<>(), device_,
                              static_cast<int64_t>(scheduledBatch) * static_cast<int64_t>(sizeof(int64_t)), &stream);
    cudaPipe_->sampledReady.record(stream);
  }

post_forward:
  prev_.seqs.clear();
  prev_.seqs.reserve(static_cast<size_t>(scheduledBatch));
  for (int32_t si = 0; si < scheduledBatch; si++) {
    const int32_t ai = scheduled[static_cast<size_t>(si)];
    prev_.seqs.push_back(active[static_cast<size_t>(ai)]);
  }
  prev_.scheduledBatch = scheduledBatch;
  prev_.valid = true;

  // process streaming callbacks after launching GPU work (overlaps with forward)
  processCallbacks();

  sweepFinished(active);
}

// sync previous step's sampling, read token IDs, update sequence state.
void Scheduler::harvestTokenIds() {
  if (!prev_.valid) {
    return;
  }

  if (cudaPipe_->eventInit) {
    CUDA_CHECK(cudaEventSynchronize(cudaPipe_->sampledReady.event()));
  }

  const int64_t* sampledH = sampledHost_.dataPtr<int64_t>();
  pendingCallbacks_.clear();

  for (int32_t si = 0; si < prev_.scheduledBatch; si++) {
    Seq& s = *prev_.seqs[static_cast<size_t>(si)];
    if (s.finished) {
      continue;
    }

    const auto tokenId = static_cast<int32_t>(sampledH[si]);
    s.lastToken = tokenId;

    if (s.numPromptProcessed != static_cast<int32_t>(s.promptIds.size())) {
      continue;
    }

    if (isEos(s, tokenId)) {
      s.finishReason = FinishReason::Stop;
      s.finished = true;
      continue;
    }

    s.generatedIds.push_back(tokenId);

    // defer callback to processCallbacks() — runs while GPU is busy
    if (s.onToken) {
      pendingCallbacks_.push_back({prev_.seqs[static_cast<size_t>(si)], tokenId});
    }

    if (!s.finished && static_cast<int32_t>(s.generatedIds.size()) >= s.maxNewTokens) {
      s.finishReason = FinishReason::Length;
      s.finished = true;
    }
  }

  prev_.seqs.clear();
  prev_.scheduledBatch = 0;
  prev_.valid = false;
}

// process deferred streaming callbacks (detokenize + onToken), overlaps with GPU.
void Scheduler::processCallbacks() {
  for (auto& [seq, tokenId] : pendingCallbacks_) {
    Seq& s = *seq;
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

// pad metadata buffers so totalTokens matches graphBatch (fills host buffers only).
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

  auto* i64Host = hostI64_.dataPtr<int64_t>();
  auto* i32Host = hostI32_.dataPtr<int32_t>();

  // fill padding entries [realBatch, graphBatch)
  for (int32_t pi = 0; pi < padCount; pi++) {
    const int32_t si = realBatch + pi;
    const int32_t ti = realTokens + pi;

    i64Host[offTokens + ti] = 0;
    i64Host[offPositions + ti] = 0;
    i64Host[offLastIdx + si] = ti;

    i32Host[offSlot + ti] = dummyBlockId_ * pageSize;
    i32Host[offCuQ + si + 1] = i32Host[offCuQ + si] + 1;
    i32Host[offCuKv + si + 1] = i32Host[offCuKv + si] + 1;

    int32_t* btRow = i32Host + offBlockTable + static_cast<size_t>(si) * btStride;
    btRow[0] = dummyBlockId_;
    std::fill(btRow + 1, btRow + btStride, -1);
  }
}

void Scheduler::workerLoop() {
  std::vector<std::shared_ptr<Seq>> active;

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
  sweepFinished(active);

  runningCount_.store(0);
  for (auto& s : active) {
    s->finishReason = FinishReason::Aborted;
    completeSeq(s);
  }
}

}  // namespace tinygpt
