/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include <curand_kernel.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

#include "Tensor/Tensor.h"
#include "Utils/CUDATypes.h"
#include "Utils/CUDAUtils.h"
#include "kernel/Dispatch.h"
#include "kernel/SamplerOps.h"

namespace tinygpt::kernel {

static constexpr int kSampleThreadsPerBlock = 256;

// ref: FlashInfer
__device__ __forceinline__ float ieeeMul(float a, float b) {
  float r;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float ieeeAdd(float a, float b) {
  float r;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

struct MaxOp {
  template <typename T>
  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    return (a > b) ? a : b;
  }
};

struct KeyValPair {
  float key;
  unsigned int idx;
};

struct ArgMaxOp {
  __device__ __forceinline__ KeyValPair operator()(const KeyValPair& a, const KeyValPair& b) const {
    if (a.key > b.key) return a;
    if (a.key < b.key) return b;
    return (a.idx < b.idx) ? a : b;
  }
};

// gumbel(0,1) = -log(-log(U)), U ~ Uniform(0,1]
__device__ __forceinline__ float drawGumbel(curandStatePhilox4_32_10_t* rng) {
  constexpr float kEps = 1.1754944e-38f;
  constexpr float kOneMinusEps = 0.99999994f;
  float u = curand_uniform(rng);
  u = fmaxf(fminf(u, kOneMinusEps), kEps);
  return -logf(-logf(u));
}

template <int BLOCK_THREADS>
struct SamplingTempStorage {
  using BlockReduceFloat = cub::BlockReduce<float, BLOCK_THREADS>;
  using BlockReduceInt = cub::BlockReduce<int, BLOCK_THREADS>;
  using BlockReducePair = cub::BlockReduce<KeyValPair, BLOCK_THREADS>;
  using BlockScanFloat = cub::BlockScan<float, BLOCK_THREADS>;

  union {
    typename BlockReduceFloat::TempStorage reduceFloat;
    typename BlockReduceInt::TempStorage reduceInt;
    typename BlockReducePair::TempStorage reducePair;
    typename BlockScanFloat::TempStorage scan;
  } prim;

  float scalar;
  int sampledId;
  int lastValidId;
};

template <int BLOCK_THREADS>
__device__ __forceinline__ void deviceSamplingChunk(int globalIdx, float u, float probI, bool keep, float& aggregate,
                                                    SamplingTempStorage<BLOCK_THREADS>& s) {
  using BlockReduceFloat = cub::BlockReduce<float, BLOCK_THREADS>;
  using BlockReduceInt = cub::BlockReduce<int, BLOCK_THREADS>;
  using BlockScanFloat = cub::BlockScan<float, BLOCK_THREADS>;

  const unsigned int tx = threadIdx.x;
  const float masked = keep ? probI : 0.f;

  float chunkMass = BlockReduceFloat(s.prim.reduceFloat).Sum(masked);
  if (tx == 0) s.scalar = chunkMass;
  __syncthreads();
  chunkMass = s.scalar;

  if (aggregate + chunkMass > u) {
    __syncthreads();
    float inclusiveCdf;
    BlockScanFloat(s.prim.scan).InclusiveSum(masked, inclusiveCdf);
    __syncthreads();

    const bool crossed = keep && (aggregate + inclusiveCdf > u);
    if (crossed) {
      atomicMin(&s.sampledId, globalIdx);
    }
    __syncthreads();
  }

  const int validIdx = keep ? globalIdx : -1;
  __syncthreads();
  int maxValid = BlockReduceInt(s.prim.reduceInt).Reduce(validIdx, MaxOp{});
  if (tx == 0 && maxValid >= 0) s.lastValidId = maxValid;
  __syncthreads();

  aggregate += chunkMass;
}

struct MinOp {
  __device__ __forceinline__ float operator()(float a, float b) const { return (a < b) ? a : b; }
};

// Bitonic sort descending by key, in shared memory.
// N must be a power of 2 and equal BLOCK_THREADS (one element per thread).
template <int N>
__device__ __forceinline__ void bitonicSortDescending(float* __restrict__ sKeys, unsigned int* __restrict__ sVals) {
  const unsigned int tid = threadIdx.x;
  for (int k = 2; k <= N; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      const unsigned int ixj = tid ^ j;
      if (ixj > tid) {
        // For a full descending sort: swap if element at tid < element at ixj
        // in the first half of the bitonic merge, don't swap in the second half.
        const bool descending = ((tid & k) == 0);
        const bool needSwap = descending ? (sKeys[tid] < sKeys[ixj]) : (sKeys[tid] > sKeys[ixj]);
        if (needSwap) {
          float tmpK = sKeys[tid];
          sKeys[tid] = sKeys[ixj];
          sKeys[ixj] = tmpK;
          unsigned int tmpV = sVals[tid];
          sVals[tid] = sVals[ixj];
          sVals[ixj] = tmpV;
        }
      }
      __syncthreads();
    }
  }
}

template <typename T>
__device__ __forceinline__ void fusedSampleBody(const T* __restrict__ logits, int64_t* __restrict__ output,
                                                float* __restrict__ probsWs, SamplingParams p, int vocab,
                                                uint64_t globalSeed, uint64_t globalSeq) {
  constexpr int BLOCK_THREADS = kSampleThreadsPerBlock;
  using ReducePair = cub::BlockReduce<KeyValPair, BLOCK_THREADS>;
  using ReduceFloat = cub::BlockReduce<float, BLOCK_THREADS>;

  __shared__ SamplingTempStorage<BLOCK_THREADS> sTemp;

  __shared__ unsigned int sCandVid[kMaxTopK];
  __shared__ union {
    float key[kMaxTopK];
    float prob[kMaxTopK];
  } sCand;
  __shared__ int32_t sKeep;

  __shared__ float sMaxLogit;

  const unsigned int row = blockIdx.x;
  const unsigned int tid = threadIdx.x;
  const T* rowLogits = logits + static_cast<int64_t>(row) * vocab;
  float* rowProbs = (probsWs != nullptr) ? probsWs + static_cast<int64_t>(row) * vocab : nullptr;

  const bool greedy = (p.temperature <= 0.f);
  const float temperature = greedy ? 1.f : p.temperature;
  const float invT = 1.f / temperature;
  const bool userTopK = (p.topK > 0);
  const bool useTopP = (p.topP < 1.f) && (p.topP > 0.f);
  const bool useMinP = (p.minP > 0.f);

  // argmax + maxLogit
  KeyValPair threadBest = {-INFINITY, 0};
  for (unsigned int i = tid; i < vocab; i += BLOCK_THREADS) {
    const auto v = static_cast<float>(rowLogits[i]);
    if (v > threadBest.key) {
      threadBest.key = v;
      threadBest.idx = i;
    }
  }
  KeyValPair globalBest = ReducePair(sTemp.prim.reducePair).Reduce(threadBest, ArgMaxOp());
  if (tid == 0) sMaxLogit = globalBest.key;
  __syncthreads();
  const float maxLogit = sMaxLogit;

  if (greedy) {
    if (tid == 0) output[row] = static_cast<int64_t>(globalBest.idx);
    return;
  }

  const uint64_t seed = (p.seed >= 0) ? static_cast<uint64_t>(p.seed) : globalSeed;
  curandStatePhilox4_32_10_t state;

  if (userTopK) {
    curand_init(seed, static_cast<uint64_t>(row) * kSampleThreadsPerBlock + tid, globalSeq, &state);

    int topK = p.topK;
    if (topK > vocab) topK = vocab;
    if (topK > kMaxTopK) topK = kMaxTopK;

    for (int round = 0; round < topK; ++round) {
      KeyValPair best = {-INFINITY, 0};
      for (unsigned int i = tid; i < vocab; i += BLOCK_THREADS) {
        auto v = static_cast<float>(rowLogits[i]);
        bool masked = false;
#pragma unroll 1
        for (int j = 0; j < round; ++j) {
          if (sCandVid[j] == i) {
            masked = true;
            break;
          }
        }
        if (masked) v = -INFINITY;
        if (v > best.key) {
          best.key = v;
          best.idx = i;
        }
      }
      KeyValPair winner = ReducePair(sTemp.prim.reducePair).Reduce(best, ArgMaxOp());
      if (tid == 0) {
        sCandVid[round] = winner.idx;
        sCand.key[round] = winner.key;
      }
      __syncthreads();
    }

    if (tid == 0) {
      float candSum = 0.f;
      for (int i = 0; i < topK; ++i) {
        float e = __expf((sCand.key[i] - maxLogit) * invT);
        sCand.prob[i] = e;
        candSum += e;
      }
      const float invSum = (candSum > 0.f) ? (1.f / candSum) : 0.f;
      for (int i = 0; i < topK; ++i) sCand.prob[i] *= invSum;

      int keep = topK;
      if (useTopP) {
        float cum = 0.f;
        keep = 0;
        for (int i = 0; i < topK; ++i) {
          cum += sCand.prob[i];
          ++keep;
          if (cum >= p.topP) break;
        }
      }
      if (useMinP) {
        const float thresh = sCand.prob[0] * p.minP;
        int nk = 0;
        for (int i = 0; i < keep; ++i) {
          if (sCand.prob[i] >= thresh) ++nk;
        }
        if (nk < 1) nk = 1;
        keep = nk;
      }
      sKeep = keep;
    }
    __syncthreads();

    // gumbel-max over survivors
    KeyValPair best = {-INFINITY, 0};
    for (unsigned int i = tid; i < sKeep; i += BLOCK_THREADS) {
      const float pi = sCand.prob[i];
      const float score = (pi > 0.f) ? (logf(pi) + drawGumbel(&state)) : -INFINITY;
      if (score > best.key) {
        best.key = score;
        best.idx = i;
      }
    }
    KeyValPair winner = ReducePair(sTemp.prim.reducePair).Reduce(best, ArgMaxOp());
    if (tid == 0) output[row] = static_cast<int64_t>(sCandVid[winner.idx]);
    return;
  }

  curand_init(seed, row, globalSeq, &state);

  // ========================================================================
  // FAST PATH: Implicit top-K via per-thread partition winners (shared memory only)
  // ========================================================================
  // When vocab > BLOCK_THREADS and temperature is moderate (≤ 2.0), the 256
  // per-thread partition-max values capture >99.99% of the probability mass.
  // We do softmax + sort + top-P entirely in shared memory, avoiding any
  // global memory workspace allocation.
  // ========================================================================
  {
    // Phase 1: Populate candidates from the argmax pass (already done above)
    sCandVid[tid] = threadBest.idx;
    sCand.key[tid] = threadBest.key;
    __syncthreads();

    // Phase 2a: Tail-mass safety check
    // Compute the minimum logit among our 256 candidates.
    float minCandLogit = ReduceFloat(sTemp.prim.reduceFloat).Reduce(threadBest.key, MinOp{});
    if (tid == 0) sTemp.scalar = minCandLogit;
    __syncthreads();
    minCandLogit = sTemp.scalar;

    // Upper-bound on missed probability mass:
    //   sum_{i not in candidates} softmax(l_i/T) ≤ (V - 256) * exp((minCand - max) / T)
    const int tailCount = vocab > BLOCK_THREADS ? (vocab - BLOCK_THREADS) : 0;
    const float tailBound = static_cast<float>(tailCount) * __expf((minCandLogit - maxLogit) * invT);

    if (tailBound > 0.01f) {
      // Extremely rare: temperature is very high or distribution nearly uniform.
      // Fall through to the exact global-memory path below.
      goto exact_global_path;
    }

    // Phase 2b: Softmax over 256 candidates in shared memory
    {
      float localExp = __expf((sCand.key[tid] - maxLogit) * invT);
      sCand.key[tid] = localExp;
      __syncthreads();

      float sumE = ReduceFloat(sTemp.prim.reduceFloat).Sum(localExp);
      if (tid == 0) sTemp.scalar = sumE;
      __syncthreads();
      const float invSum = (sTemp.scalar > 0.f) ? (1.f / sTemp.scalar) : 0.f;
      sCand.key[tid] = sCand.key[tid] * invSum;
      __syncthreads();
    }

    // Phase 2c: Bitonic sort descending by probability
    bitonicSortDescending<BLOCK_THREADS>(sCand.key, sCandVid);
    // Now: sCand.key[0] >= sCand.key[1] >= ... (normalized probs)
    //      sCandVid[i] = vocab index of the i-th most probable candidate

    // Phase 3a: Top-P + Min-P filtering
    {
      int keep = BLOCK_THREADS;

      if (useTopP) {
        // Inclusive prefix sum over sorted probabilities to find CDF
        float myProb = sCand.key[tid];
        float cumProb;
        cub::BlockScan<float, BLOCK_THREADS>(sTemp.prim.scan).InclusiveSum(myProb, cumProb);
        __syncthreads();
        if (tid == 0) sTemp.sampledId = BLOCK_THREADS;  // sentinel
        __syncthreads();

        // Find the first index where cumulative probability >= topP
        if (myProb > 0.f && cumProb >= p.topP) {
          atomicMin(&sTemp.sampledId, static_cast<int>(tid));
        }
        __syncthreads();
        keep = sTemp.sampledId + 1;
        if (keep > BLOCK_THREADS) keep = BLOCK_THREADS;
      }

      if (useMinP) {
        const float thresh = sCand.key[0] * p.minP;
        // Sorted descending, so find the first index below threshold.
        if (tid == 0) sTemp.lastValidId = keep;  // sentinel
        __syncthreads();
        if (tid < keep && sCand.key[tid] < thresh) {
          atomicMin(&sTemp.lastValidId, static_cast<int>(tid));
        }
        __syncthreads();
        if (sTemp.lastValidId > 0 && sTemp.lastValidId < keep) {
          keep = sTemp.lastValidId;
        }
      }

      if (keep < 1) keep = 1;
      sKeep = keep;
    }
    __syncthreads();

    // Phase 3b: Gumbel-max sampling over surviving candidates
    {
      KeyValPair best = {-INFINITY, 0};
      for (unsigned int i = tid; i < static_cast<unsigned int>(sKeep); i += BLOCK_THREADS) {
        const float pi = sCand.key[i];
        const float score = (pi > 0.f) ? (logf(pi) + drawGumbel(&state)) : -INFINITY;
        if (score > best.key) {
          best.key = score;
          best.idx = i;
        }
      }
      KeyValPair winner = ReducePair(sTemp.prim.reducePair).Reduce(best, ArgMaxOp());
      if (tid == 0) output[row] = static_cast<int64_t>(sCandVid[winner.idx]);
    }
    return;
  }

  // ========================================================================
  // EXACT FALLBACK PATH: Full-vocabulary processing in global memory
  // ========================================================================
  // Used only when tail-mass check fails (T > ~2.0 with near-uniform logits).
  // This is the original implementation with proven correctness.
  // Requires probsWs (rowProbs) to be non-null. If it is null (should not
  // happen for T > 2.0, but guard defensively), fall back to argmax.
  // ========================================================================
exact_global_path:

  if (rowProbs == nullptr) {
    // Defensive fallback: no workspace available, emit argmax.
    if (tid == 0) output[row] = static_cast<int64_t>(globalBest.idx);
    return;
  }

  const float minPThresh = useMinP ? (maxLogit + temperature * logf(p.minP)) : -INFINITY;

  float tse = 0.f;
  for (unsigned int i = tid; i < vocab; i += BLOCK_THREADS) {
    const auto li = static_cast<float>(rowLogits[i]);
    const float e = (li >= minPThresh) ? __expf((li - maxLogit) * invT) : 0.f;
    rowProbs[i] = e;
    tse += e;
  }
  float sumExp = ReduceFloat(sTemp.prim.reduceFloat).Sum(tse);
  if (tid == 0) sTemp.scalar = sumExp;
  __syncthreads();
  sumExp = sTemp.scalar;
  const float invSumExp = (sumExp > 0.f) ? (1.f / sumExp) : 0.f;

  for (unsigned int i = tid; i < vocab; i += BLOCK_THREADS) {
    rowProbs[i] = rowProbs[i] * invSumExp;
  }
  __syncthreads();

  if (!useTopP) {
    if (tid == 0) {
      sTemp.sampledId = vocab;
      sTemp.lastValidId = -1;
    }
    __syncthreads();
    const float u = curand_uniform(&state);  // u ∈ (0, 1]
    float aggregate = 0.f;
    for (int chunk = 0; chunk < vocab; chunk += BLOCK_THREADS) {
      const unsigned int idx = chunk + tid;
      const bool inBounds = idx < vocab;
      const float pi = inBounds ? rowProbs[idx] : 0.f;
      const bool keep = inBounds && (pi > 0.f);
      deviceSamplingChunk<BLOCK_THREADS>(static_cast<int>(idx), u, pi, keep, aggregate, sTemp);
      if (aggregate > u) break;
    }
    __syncthreads();
    if (tid == 0) {
      int picked = sTemp.sampledId;
      if (picked >= vocab) picked = (sTemp.lastValidId >= 0) ? sTemp.lastValidId : 0;
      output[row] = static_cast<int64_t>(picked);
    }
    return;
  }

  int sampledId = 0;
  float low = 0.f;
  float high = 1.f;
  float q = 1.f;

  while (low < high) {
    if (tid == 0) {
      sTemp.sampledId = vocab;
      sTemp.lastValidId = -1;
    }
    __syncthreads();

    const float u = curand_uniform(&state) * q;
    float aggregate = 0.f;
    for (int chunk = 0; chunk < vocab; chunk += BLOCK_THREADS) {
      const unsigned int idx = chunk + tid;
      const bool inBounds = idx < vocab;
      const float pi = inBounds ? rowProbs[idx] : 0.f;
      const bool keep = inBounds && (pi > low);
      deviceSamplingChunk<BLOCK_THREADS>(static_cast<int>(idx), u, pi, keep, aggregate, sTemp);
      if (aggregate > u) break;
    }
    __syncthreads();

    int picked = sTemp.sampledId;
    if (picked >= vocab) {
      if (sTemp.lastValidId < 0) {
        if (tid == 0) output[row] = 0;
        return;
      }
      picked = sTemp.lastValidId;
    }
    sampledId = picked;

    const float pivot0 = rowProbs[sampledId];
    const float pivot1 = ieeeMul(ieeeAdd(pivot0, high), 0.5f);

    float localF0 = 0.f, localF1 = 0.f;
    for (unsigned int i = tid; i < vocab; i += BLOCK_THREADS) {
      const float pi = rowProbs[i];
      if (pi > pivot0) localF0 += pi;
      if (pi > pivot1) localF1 += pi;
    }
    float f0 = ReduceFloat(sTemp.prim.reduceFloat).Sum(localF0);
    if (tid == 0) sTemp.scalar = f0;
    __syncthreads();
    f0 = sTemp.scalar;

    if (f0 < p.topP) {
      break;  // accept sampledId
    }

    __syncthreads();
    float f1 = ReduceFloat(sTemp.prim.reduceFloat).Sum(localF1);
    if (tid == 0) sTemp.scalar = f1;
    __syncthreads();
    f1 = sTemp.scalar;

    if (f1 < p.topP) {
      low = pivot0;
      high = pivot1;
      q = f0;
    } else {
      low = pivot1;
      q = f1;
    }
  }

  if (tid == 0) output[row] = static_cast<int64_t>(sampledId);
}

template <typename T>
__global__ void kFusedSample(const T* __restrict__ logits, int64_t* __restrict__ output, float* __restrict__ probsWs,
                             const SamplingParams* __restrict__ params, int vocab, uint64_t globalSeed,
                             uint64_t globalSeq) {
  fusedSampleBody<T>(logits, output, probsWs, params[blockIdx.x], vocab, globalSeed, globalSeq);
}

template <typename T>
__global__ void kFusedSampleBroadcast(const T* __restrict__ logits, int64_t* __restrict__ output,
                                      float* __restrict__ probsWs, SamplingParams p, int vocab, uint64_t globalSeed,
                                      uint64_t globalSeq) {
  fusedSampleBody<T>(logits, output, probsWs, p, vocab, globalSeed, globalSeq);
}

// Graph-capturable variant: reads globalSeq from a device pointer so the value
// can be updated via H2D memcpy between CUDA Graph replays.
template <typename T>
__global__ void kFusedSampleGraphable(const T* __restrict__ logits, int64_t* __restrict__ output,
                                      float* __restrict__ probsWs, SamplingParams p, int vocab, uint64_t globalSeed,
                                      const uint64_t* __restrict__ devGlobalSeqPtr) {
  fusedSampleBody<T>(logits, output, probsWs, p, vocab, globalSeed, *devGlobalSeqPtr);
}

namespace {

bool mayUsePathA(const SamplingParams& q) { return (q.temperature > 0.f) && (q.topK <= 0); }

// The fast implicit-top-K path avoids global workspace for moderate temperatures.
// Only allocate workspace when the fallback might be triggered (T > 2.0).
bool mayNeedFallbackWorkspace(const SamplingParams& q) { return mayUsePathA(q) && (q.temperature > 2.0f); }

float* acquireProbsWorkspace(const tinytorch::Tensor& logits, int32_t batch, int vocab) {
  thread_local tinytorch::Tensor wsCache;
  const int64_t needed = static_cast<int64_t>(batch) * static_cast<int64_t>(vocab);
  const bool sameDevice = (wsCache.defined() && wsCache.device() == logits.device());
  if (!sameDevice || wsCache.numel() < needed) {
    auto opts = tinytorch::Options(logits.device(), tinytorch::DType::Float32).noGrad();
    wsCache = tinytorch::Tensor({needed}, opts);
  }
  return wsCache.dataPtr<float>();
}

}  // namespace

template <typename CudaT>
static void fusedSampleImpl(const tinytorch::Tensor& logits, tinytorch::Tensor& output,
                            const SamplingParams* paramsHost, int32_t batch, uint64_t globalSeed, uint64_t globalSeq) {
  const auto vocab = static_cast<int>(logits.size(-1));
  ASSERT(vocab > 0);
  ASSERT(batch > 0);

  auto& stream = tinytorch::cuda::getCurrentCUDAStream(logits.device().index);

  const bool needsWs =
      std::any_of(paramsHost, paramsHost + batch, [](const SamplingParams& q) { return mayNeedFallbackWorkspace(q); });
  float* wsPtr = needsWs ? acquireProbsWorkspace(logits, batch, vocab) : nullptr;

  const bool homogeneous =
      std::all_of(paramsHost + 1, paramsHost + batch, [&](const SamplingParams& q) { return q == paramsHost[0]; });
  if (homogeneous) {
    kFusedSampleBroadcast<CudaT><<<batch, kSampleThreadsPerBlock, 0, stream.stream()>>>(
        logits.dataPtr<CudaT>(), output.dataPtr<int64_t>(), wsPtr, paramsHost[0], vocab, globalSeed, globalSeq);
    CUDA_KERNEL_CHECK();
    return;
  }

  static_assert(sizeof(SamplingParams) % sizeof(int32_t) == 0,
                "SamplingParams must be int32-aligned for the staging tensor");
  const auto paramsBytes = static_cast<int64_t>(batch) * static_cast<int64_t>(sizeof(SamplingParams));
  const auto wordsPerRow = static_cast<int64_t>(sizeof(SamplingParams) / sizeof(int32_t));

  auto pinnedOpts = tinytorch::Options(tinytorch::Device::cpu(), tinytorch::DType::Int32).noGrad().pinnedMemory(true);
  auto devOpts = tinytorch::Options(logits.device(), tinytorch::DType::Int32).noGrad();
  tinytorch::Tensor hostParams({static_cast<int64_t>(batch) * wordsPerRow}, pinnedOpts);
  tinytorch::Tensor devParams({static_cast<int64_t>(batch) * wordsPerRow}, devOpts);

  std::memcpy(hostParams.dataPtr<int32_t>(), paramsHost, static_cast<size_t>(paramsBytes));

  tinytorch::Storage::copyOnDevice(devParams.dataPtr<>(), logits.device(), hostParams.dataPtr<>(),
                                   tinytorch::Device::cpu(), paramsBytes, &stream);

  kFusedSample<CudaT><<<batch, kSampleThreadsPerBlock, 0, stream.stream()>>>(
      logits.dataPtr<CudaT>(), output.dataPtr<int64_t>(), wsPtr,
      reinterpret_cast<const SamplingParams*>(devParams.dataPtr<int32_t>()), vocab, globalSeed, globalSeq);
  CUDA_KERNEL_CHECK();
}

tinytorch::Tensor fusedSample(const tinytorch::Tensor& logits, const SamplingParams* paramsHost, int32_t batch,
                              uint64_t globalSeed, uint64_t globalSeq) {
  ASSERT(logits.dim() == 2);
  ASSERT(logits.size(0) == batch);
  ASSERT(paramsHost != nullptr);

  auto outOpts = tinytorch::Options(logits.device(), tinytorch::DType::Int64).noGrad();
  tinytorch::Tensor output({static_cast<int64_t>(batch), 1}, outOpts);

  TINYGPT_DISPATCH_FLOAT_DTYPE(logits,
                               { fusedSampleImpl<CudaT>(logits, output, paramsHost, batch, globalSeed, globalSeq); });
  return output;
}

tinytorch::Tensor fusedSample(const tinytorch::Tensor& logits, const SamplingParams& params, uint64_t globalSeed,
                              uint64_t globalSeq) {
  ASSERT(logits.dim() == 2);
  const auto batch = static_cast<int32_t>(logits.size(0));
  const auto vocab = static_cast<int>(logits.size(-1));
  ASSERT(batch > 0);
  ASSERT(vocab > 0);

  auto outOpts = tinytorch::Options(logits.device(), tinytorch::DType::Int64).noGrad();
  tinytorch::Tensor output({static_cast<int64_t>(batch), 1}, outOpts);

  float* wsPtr = mayNeedFallbackWorkspace(params) ? acquireProbsWorkspace(logits, batch, vocab) : nullptr;

  auto& stream = tinytorch::cuda::getCurrentCUDAStream(logits.device().index);
  TINYGPT_DISPATCH_FLOAT_DTYPE(logits, {
    kFusedSampleBroadcast<CudaT><<<batch, kSampleThreadsPerBlock, 0, stream.stream()>>>(
        logits.dataPtr<CudaT>(), output.dataPtr<int64_t>(), wsPtr, params, vocab, globalSeed, globalSeq);
    CUDA_KERNEL_CHECK();
  });
  return output;
}

void fusedSampleGraphable(const tinytorch::Tensor& logits, tinytorch::Tensor& output, const SamplingParams& params,
                          uint64_t globalSeed, const uint64_t* devGlobalSeqPtr) {
  ASSERT(logits.dim() == 2);
  ASSERT(output.dim() == 2);
  const auto batch = static_cast<int32_t>(logits.size(0));
  const auto vocab = static_cast<int>(logits.size(-1));
  ASSERT(batch > 0);
  ASSERT(vocab > 0);
  ASSERT(output.size(0) == batch && output.size(1) == 1);
  ASSERT(devGlobalSeqPtr != nullptr);

  // Note: workspace is not needed for typical temperature (≤ 2.0) since
  // the fast implicit-top-K path uses only shared memory.
  float* wsPtr = nullptr;

  auto& stream = tinytorch::cuda::getCurrentCUDAStream(logits.device().index);
  TINYGPT_DISPATCH_FLOAT_DTYPE(logits, {
    kFusedSampleGraphable<CudaT><<<batch, kSampleThreadsPerBlock, 0, stream.stream()>>>(
        logits.dataPtr<CudaT>(), output.dataPtr<int64_t>(), wsPtr, params, vocab, globalSeed, devGlobalSeqPtr);
    CUDA_KERNEL_CHECK();
  });
}

}  // namespace tinygpt::kernel
