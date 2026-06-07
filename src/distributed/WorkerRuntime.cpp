/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "WorkerRuntime.h"

#include <algorithm>

#include "Communicator.h"
#include "Tensor/CachedAllocator.h"
#include "Utils/CUDAUtils.h"
#include "Utils/Logger.h"
#include "engine/ForwardContext.h"
#include "engine/PagedKVCache.h"
#include "engine/Scheduler.h"
#include "kernel/AttentionOps.h"
#include "model/GPTModel.h"

namespace tt = tinytorch;

namespace tinygpt::distributed {

WorkerRuntime::WorkerRuntime(GPTModel& model, PagedKVCache& cache, int32_t maxBatchTokens)
    : model_(model), cache_(cache), maxBatchTokens_(maxBatchTokens), device_(model.device()) {
  ASSERT(device_.isCuda());

  constexpr int32_t kMinPartSize = kernel::kSplitKvMinPartitionSize;
  constexpr int32_t kMaxPartitions = kernel::kSplitKvMaxPartitions;
  const int32_t maxSeqLenKV = cache_.maxBlocksPerSeq() * cache_.blockSize();
  const int32_t numPartitions = std::min((maxSeqLenKV + kMinPartSize - 1) / kMinPartSize, kMaxPartitions);

  const int64_t localHeads = model_.numHeads() / Communicator::tp().worldSize();

  auto floatOpts = tt::Options(device_, tt::DType::Float32).noGrad();
  const int64_t splitOSize = static_cast<int64_t>(maxBatchTokens_) * localHeads * numPartitions * model_.headDim();
  const int64_t splitLseSize = static_cast<int64_t>(maxBatchTokens_) * localHeads * numPartitions * 2;
  splitKvO_ = tt::Tensor({splitOSize}, floatOpts);
  splitKvLse_ = tt::Tensor({splitLseSize}, floatOpts);

  allocateMetaBuffers();
}

void WorkerRuntime::allocateMetaBuffers() {
  const int64_t maxTok = maxBatchTokens_;
  const int64_t blocksPerSeq = cache_.maxBlocksPerSeq();

  const int64_t i64Total = 3 * maxTok;
  const int64_t i32Total = maxTok + (maxTok + 1) + (maxTok + 1) + maxTok * blocksPerSeq;

  auto pinnedI64 = tt::Options(tt::Device::cpu(), tt::DType::Int64).noGrad().pinnedMemory(true);
  auto devI32 = tt::Options(device_, tt::DType::Int32).noGrad();
  auto devI64 = tt::Options(device_, tt::DType::Int64).noGrad();

  metaDevI64_ = tt::Tensor({i64Total}, devI64);
  metaDevI32_ = tt::Tensor({i32Total}, devI32);

  tpHeaderDev_ = tt::Tensor({Scheduler::kTPHeaderSize}, devI64);
  tpHeaderHost_ = tt::Tensor({Scheduler::kTPHeaderSize}, pinnedI64);
}

void WorkerRuntime::run() {
  tt::NoGradGuard noGrad;
  auto& comm = Communicator::tp();
  auto& stream = tt::cuda::getCurrentCUDAStream(device_.index);

  LOGI("WorkerRuntime: rank=%d worldSize=%d entering forward loop", comm.rank(), comm.worldSize());

  while (true) {
    // receive control header from rank0
    comm.broadcast(tpHeaderDev_);
    tt::Storage::copyOnDevice(tpHeaderHost_.dataPtr<>(), tt::Device::cpu(), tpHeaderDev_.dataPtr<>(), device_,
                              Scheduler::kTPHeaderSize * static_cast<int64_t>(sizeof(int64_t)), &stream);
    stream.synchronize();
    auto* hdr = tpHeaderHost_.dataPtr<int64_t>();
    const int64_t command = hdr[0];
    if (command == 0) break;

    const int64_t totalTokens = hdr[1];
    const int64_t scheduledBatch = hdr[2];
    const int64_t maxSeqLenQ = hdr[3];
    const int64_t maxSeqLenKV = hdr[4];
    const int64_t i64Used = hdr[5];
    const int64_t i32Used = hdr[6];

    // receive metadata buffers
    auto i64View = metaDevI64_.narrow(0, 0, i64Used);
    auto i32View = metaDevI32_.narrow(0, 0, i32Used);
    comm.broadcast(i64View);
    comm.broadcast(i32View);

    const int64_t offTokens = 0;
    const int64_t offPositions = totalTokens;
    const int64_t offLastIdx = totalTokens * 2;
    const int64_t offSlot = 0;
    const int64_t offCuQ = totalTokens;
    const int64_t offCuKv = offCuQ + scheduledBatch + 1;
    const int64_t offBlockTable = offCuKv + scheduledBatch + 1;
    const int32_t pageSize = cache_.blockSize();
    const int32_t blocksPerSeq = cache_.maxBlocksPerSeq();

    auto tokensTensor = metaDevI64_.narrow(0, offTokens, totalTokens);
    auto positionsDev = metaDevI64_.narrow(0, offPositions, totalTokens);
    auto lastIdxTensor = metaDevI64_.narrow(0, offLastIdx, scheduledBatch);
    auto slotDev = metaDevI32_.narrow(0, offSlot, totalTokens);
    auto cuQDev = metaDevI32_.narrow(0, offCuQ, scheduledBatch + 1);
    auto cuKvDev = metaDevI32_.narrow(0, offCuKv, scheduledBatch + 1);
    auto blockTableDev = metaDevI32_.narrow(0, offBlockTable, scheduledBatch * blocksPerSeq)
                             .view({static_cast<int64_t>(scheduledBatch), blocksPerSeq});

    ForwardContext ctx;
    ctx.pagedCache = &cache_;
    ctx.positions = positionsDev;
    ctx.slotMapping = slotDev;
    ctx.cuSeqLensQ = cuQDev;
    ctx.cuSeqLensKV = cuKvDev;
    ctx.blockTable = blockTableDev;
    ctx.maxSeqLenQ = static_cast<int32_t>(maxSeqLenQ);
    ctx.maxSeqLenKV = static_cast<int32_t>(maxSeqLenKV);
    ctx.pageSize = pageSize;
    ctx.maxBlocksPerSeq = blocksPerSeq;
    ctx.tmpO = splitKvO_.dataPtr<float>();
    ctx.tmpLse = splitKvLse_.dataPtr<float>();
    ctx.lastTokenIndices = lastIdxTensor;

    {
      ForwardContextGuard guard(&ctx);
      // workers don't need the logits
      auto logits = model_.forward(tokensTensor);
      UNUSED(logits);
    }
  }

  LOGI("WorkerRuntime: rank=%d exiting", comm.rank());
}

}  // namespace tinygpt::distributed
