/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "PagedKVCache.h"

#include <cinttypes>

#include "Utils/CUDAUtils.h"
#include "Utils/Logger.h"
#include "model/GPTModel.h"

namespace tt = tinytorch;

namespace tinygpt {

namespace {

int64_t queryFreeBytes(const tt::Device& device) {
  if (!device.isCuda()) {
    return 0;
  }
  tt::cuda::CudaDeviceGuard guard(device.index);
  size_t freeBytes = 0;
  size_t totalBytes = 0;
  CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
  LOGI("PagedKVCache: device %d  free=%.2f GiB / total=%.2f GiB", device.index,
       static_cast<double>(freeBytes) / (1024.0 * 1024.0 * 1024.0),
       static_cast<double>(totalBytes) / (1024.0 * 1024.0 * 1024.0));
  return static_cast<int64_t>(freeBytes);
}

}  // namespace

PagedKVCacheSizing PagedKVCache::autoSize(const GPTModel& model, tt::DType dtype, const PagedKVCacheConfig& cfg) {
  ASSERT(cfg.blockSize > 0);
  ASSERT(cfg.maxSeqLen > 0);
  ASSERT(model.device().isCuda());

  const auto L = model.numLayers();
  const auto Hk = model.numKvHeads();
  const auto D = model.headDim();
  const auto elemBytes = static_cast<int64_t>(tt::dtypeSize(dtype));

  const int64_t bytesPerBlock = 2LL * L * Hk * cfg.blockSize * D * elemBytes;
  ASSERT(bytesPerBlock > 0);

  auto numBlocks = cfg.numBlocks;
  if (numBlocks <= 0) {
    auto freeBytes = queryFreeBytes(model.device());
    ASSERT(freeBytes > cfg.reserveBytes && "not enough free VRAM for paged KV cache; reduce reserveBytes");
    auto avail = static_cast<int64_t>(static_cast<double>(freeBytes - cfg.reserveBytes) * cfg.memoryUtil);
    ASSERT(avail > 0);
    numBlocks = avail / bytesPerBlock;
    ASSERT(numBlocks > 0 && "auto-sized numBlocks == 0; reduce reserveBytes / blockSize / dtype");
  }

  auto maxBlocksPerSeq = static_cast<int32_t>((cfg.maxSeqLen + cfg.blockSize - 1) / cfg.blockSize);

  LOGI("PagedKVCache: sizing  blockSize=%d  numBlocks=%" PRId64 "  (%.2f GiB total KV pool)  maxBlocksPerSeq=%d",
       cfg.blockSize, numBlocks, static_cast<double>(numBlocks * bytesPerBlock) / (1024.0 * 1024.0 * 1024.0),
       maxBlocksPerSeq);

  return {numBlocks, cfg.blockSize, maxBlocksPerSeq};
}

PagedKVCache::PagedKVCache(int64_t numLayers, int64_t numKvHeads, int64_t headDim, const PagedKVCacheSizing& sizing,
                           tt::Options options)
    : numBlocks_(sizing.numBlocks),
      blockSize_(sizing.blockSize),
      maxBlocksPerSeq_(sizing.maxBlocksPerSeq),
      numLayers_(numLayers),
      numKvHeads_(numKvHeads),
      headDim_(headDim) {
  ASSERT(numBlocks_ > 0);
  ASSERT(blockSize_ > 0);

  // per-layer K/V pool: [numBlocks, numKvHeads, blockSize, headDim]
  kPool_.reserve(numLayers_);
  vPool_.reserve(numLayers_);

  const int64_t nb = numBlocks_;
  const int64_t bs = blockSize_;
  std::vector<int64_t> shape{nb, numKvHeads_, bs, headDim_};
  for (int64_t l = 0; l < numLayers_; l++) {
    kPool_.emplace_back(tt::IntArrayView(shape), options.noGrad());
    vPool_.emplace_back(tt::IntArrayView(shape), options.noGrad());
  }

  // reverse order
  freeBlocks_.reserve(static_cast<size_t>(numBlocks_));
  for (int64_t i = numBlocks_ - 1; i >= 0; i--) {
    freeBlocks_.push_back(static_cast<int32_t>(i));
  }

  LOGI("PagedKVCache: initialized  layers=%" PRId64 "  kvHeads=%" PRId64 "  headDim=%" PRId64 "  pool=%.2f GiB",
       numLayers_, numKvHeads_, headDim_, static_cast<double>(totalBytes()) / (1024.0 * 1024.0 * 1024.0));
}

int64_t PagedKVCache::totalBytes() const {
  int64_t elem = 0;
  if (!kPool_.empty()) {
    elem = static_cast<int64_t>(tt::dtypeSize(kPool_[0].dtype()));
  }
  return 2LL * numLayers_ * numBlocks_ * numKvHeads_ * blockSize_ * headDim_ * elem;
}

PagedKVCache::SeqId PagedKVCache::allocate() {
  std::lock_guard<std::mutex> lock(mutex_);
  SeqId id = nextSeqId_++;
  seqs_.emplace(id, SeqState{});
  return id;
}

void PagedKVCache::free(SeqId seqId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = seqs_.find(seqId);
  if (it == seqs_.end()) return;
  for (int32_t b : it->second.blocks) {
    freeBlocks_.push_back(b);
  }
  seqs_.erase(it);
}

bool PagedKVCache::appendTokens(SeqId seqId, int32_t numTokens, std::vector<int32_t>& outSlots) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = seqs_.find(seqId);
  ASSERT(it != seqs_.end() && "appendTokens on unknown seqId");
  auto& st = it->second;

  auto newSeqLen = static_cast<int64_t>(st.seqLen) + numTokens;
  auto needBlocks = static_cast<int32_t>((newSeqLen + blockSize_ - 1) / blockSize_);
  auto haveBlocks = static_cast<int32_t>(st.blocks.size());

  ASSERT(needBlocks <= maxBlocksPerSeq_ && "sequence exceeds maxSeqLen");

  if (needBlocks > haveBlocks) {
    int32_t toAlloc = needBlocks - haveBlocks;
    if (static_cast<int64_t>(freeBlocks_.size()) < toAlloc) {
      return false;  // OOM
    }
    for (int32_t i = 0; i < toAlloc; i++) {
      int32_t b = freeBlocks_.back();
      freeBlocks_.pop_back();
      st.blocks.push_back(b);
    }
  }

  outSlots.clear();
  outSlots.reserve(static_cast<size_t>(numTokens));
  for (int32_t i = 0; i < numTokens; i++) {
    auto tokPos = static_cast<int64_t>(st.seqLen) + i;
    auto blockIdx = tokPos / blockSize_;
    auto blockOff = static_cast<int32_t>(tokPos % blockSize_);
    auto blockId = st.blocks[static_cast<size_t>(blockIdx)];
    outSlots.push_back(blockId * blockSize_ + blockOff);
  }
  st.seqLen += numTokens;
  return true;
}

int32_t PagedKVCache::seqLen(SeqId seqId) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = seqs_.find(seqId);
  ASSERT(it != seqs_.end());
  return it->second.seqLen;
}

const std::vector<int32_t>& PagedKVCache::blocksOf(SeqId seqId) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = seqs_.find(seqId);
  ASSERT(it != seqs_.end());
  return it->second.blocks;
}

int64_t PagedKVCache::numFreeBlocks() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return static_cast<int64_t>(freeBlocks_.size());
}

}  // namespace tinygpt
