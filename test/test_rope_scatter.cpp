/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "Utils/CUDAUtils.h"
#include "Functions.h"
#include "kernel/AttentionOps.h"
#include "kernel/EmbeddingOps.h"
#include "kernel/RopeOps.h"
#include "test.h"

namespace tt = tinytorch;

namespace tinygpt {

namespace {

#define SKIP_IF_NO_CUDA()                                         \
  do {                                                            \
    if (!tt::cuda::deviceAvailable()) {                           \
      GTEST_SKIP() << "CUDA device not available; skipping test"; \
    }                                                             \
  } while (0)

tt::Tensor makeCudaTensor(const std::vector<float>& data, const std::vector<int64_t>& shape, tt::DType dtype) {
  auto cpuOpts = tt::Options(tt::Device::cpu(), tt::DType::Float32).noGrad();
  tt::Tensor host(data, shape, cpuOpts);
  return host.to(tt::Device(tt::DeviceType::CUDA)).to(dtype);
}

std::vector<float> readFloats(const tt::Tensor& tensor) {
  auto cpu = tensor.to(tt::DType::Float32).to(tt::Device::cpu());
  return cpu.toList<float>();
}

void fillRandom(std::vector<float>& v, unsigned int seed, float scale = 1.f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-scale, scale);
  for (auto& x : v) x = dist(rng);
}

inline float toBF16(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  bits += 0x7FFF + ((bits >> 16) & 1);
  bits &= 0xFFFF0000u;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

void quantizeToBF16(std::vector<float>& v) {
  for (auto& x : v) x = toBF16(x);
}

}  // namespace

// =============================================================================
// Test fused ropeScatterKVToCache
// =============================================================================

struct RopeScatterParam {
  int numTokens;
  int numKvHeads;
  int headDim;
  int blockSize;
  int numBlocks;
};

class RopeScatterFusedTest : public ::testing::TestWithParam<RopeScatterParam> {};

TEST_P(RopeScatterFusedTest, matches_separate_rope_and_scatter) {
  SKIP_IF_NO_CUDA();
  auto p = GetParam();

  const int numTokens = p.numTokens;
  const int numKvHeads = p.numKvHeads;
  const int headDim = p.headDim;
  const int blockSize = p.blockSize;
  const int numBlocks = p.numBlocks;
  const int maxPos = 4096;

  std::vector<float> keyData(numTokens * numKvHeads * headDim);
  std::vector<float> valData(numTokens * numKvHeads * headDim);
  fillRandom(keyData, 42, 0.5f);
  fillRandom(valData, 99, 0.5f);
  quantizeToBF16(keyData);
  quantizeToBF16(valData);

  std::vector<int32_t> slotData(numTokens);
  for (int i = 0; i < numTokens; i++) slotData[i] = i;

  std::vector<int64_t> posData(numTokens);
  for (int i = 0; i < numTokens; i++) posData[i] = i + 100;

  auto ropeCache = kernel::ropeInit(headDim, maxPos, 10000.f, nullptr,
                                    tt::Options(tt::Device(tt::DeviceType::CUDA), tt::DType::Float32).noGrad());

  auto keyGpu = makeCudaTensor(keyData, {numTokens, numKvHeads, headDim}, tt::DType::BFloat16);
  auto valGpu = makeCudaTensor(valData, {numTokens, numKvHeads, headDim}, tt::DType::BFloat16);

  auto slotCpuOpts = tt::Options(tt::Device::cpu(), tt::DType::Int32).noGrad();
  auto slotGpu = tt::Tensor(slotData, {numTokens}, slotCpuOpts).to(tt::Device(tt::DeviceType::CUDA));

  auto posCpuOpts = tt::Options(tt::Device::cpu(), tt::DType::Int64).noGrad();
  auto posGpu = tt::Tensor(posData, {numTokens}, posCpuOpts).to(tt::Device(tt::DeviceType::CUDA));

  auto cacheOpts = tt::Options(tt::Device(tt::DeviceType::CUDA), tt::DType::BFloat16).noGrad();
  int64_t poolSize = static_cast<int64_t>(numBlocks) * numKvHeads * blockSize * headDim;
  tt::Tensor kCacheRef = tt::Tensor::zeros({poolSize}, cacheOpts);
  tt::Tensor vCacheRef = tt::Tensor::zeros({poolSize}, cacheOpts);
  tt::Tensor kCacheFused = tt::Tensor::zeros({poolSize}, cacheOpts);
  tt::Tensor vCacheFused = tt::Tensor::zeros({poolSize}, cacheOpts);

  // Reference: separate RoPE + scatter
  auto keyRotated = kernel::ropeApply(keyGpu, ropeCache, posGpu);
  kernel::scatterKVToCache(keyRotated, valGpu, kCacheRef, vCacheRef, slotGpu, blockSize);

  // Fused
  kernel::ropeScatterKVToCache(keyGpu, valGpu, kCacheFused, vCacheFused, slotGpu, blockSize, ropeCache, posGpu);

  auto kRef = readFloats(kCacheRef);
  auto kFused = readFloats(kCacheFused);
  auto vRef = readFloats(vCacheRef);
  auto vFused = readFloats(vCacheFused);

  ASSERT_EQ(kRef.size(), kFused.size());
  ASSERT_EQ(vRef.size(), vFused.size());
  for (size_t i = 0; i < kRef.size(); i++) {
    EXPECT_EQ(kRef[i], kFused[i]) << "K cache mismatch at index " << i;
  }
  for (size_t i = 0; i < vRef.size(); i++) {
    EXPECT_EQ(vRef[i], vFused[i]) << "V cache mismatch at index " << i;
  }
}

std::string ropeScatterTestName(const ::testing::TestParamInfo<RopeScatterParam>& info) {
  auto& p = info.param;
  return "T" + std::to_string(p.numTokens) + "_H" + std::to_string(p.numKvHeads) + "_D" +
         std::to_string(p.headDim) + "_BS" + std::to_string(p.blockSize);
}

INSTANTIATE_TEST_SUITE_P(configs, RopeScatterFusedTest,
                         ::testing::Values(
                             RopeScatterParam{1, 8, 128, 16, 64},
                             RopeScatterParam{32, 8, 128, 16, 64},
                             RopeScatterParam{512, 8, 128, 16, 256},
                             RopeScatterParam{1, 32, 128, 16, 64},
                             RopeScatterParam{4, 4, 64, 16, 32},
                             RopeScatterParam{8, 8, 128, 32, 32}
                         ),
                         ropeScatterTestName);

// =============================================================================
// Test fused normRopeScatterKVToCache (K-Norm + RoPE + Scatter)
// =============================================================================

class NormRopeScatterTest : public ::testing::TestWithParam<RopeScatterParam> {};

TEST_P(NormRopeScatterTest, matches_separate_norm_rope_scatter) {
  SKIP_IF_NO_CUDA();
  auto p = GetParam();

  const int numTokens = p.numTokens;
  const int numKvHeads = p.numKvHeads;
  const int headDim = p.headDim;
  const int blockSize = p.blockSize;
  const int numBlocks = p.numBlocks;
  const int maxPos = 4096;
  const float eps = 1e-6f;

  std::vector<float> keyData(numTokens * numKvHeads * headDim);
  std::vector<float> valData(numTokens * numKvHeads * headDim);
  std::vector<float> normWeightData(headDim);
  fillRandom(keyData, 42, 0.5f);
  fillRandom(valData, 99, 0.5f);
  fillRandom(normWeightData, 77, 1.0f);
  quantizeToBF16(keyData);
  quantizeToBF16(valData);
  quantizeToBF16(normWeightData);

  std::vector<int32_t> slotData(numTokens);
  for (int i = 0; i < numTokens; i++) slotData[i] = i;

  std::vector<int64_t> posData(numTokens);
  for (int i = 0; i < numTokens; i++) posData[i] = i + 50;

  auto ropeCache = kernel::ropeInit(headDim, maxPos, 10000.f, nullptr,
                                    tt::Options(tt::Device(tt::DeviceType::CUDA), tt::DType::Float32).noGrad());

  auto keyGpu = makeCudaTensor(keyData, {numTokens, numKvHeads, headDim}, tt::DType::BFloat16);
  auto valGpu = makeCudaTensor(valData, {numTokens, numKvHeads, headDim}, tt::DType::BFloat16);
  auto normWeightGpu = makeCudaTensor(normWeightData, {headDim}, tt::DType::BFloat16);

  auto slotCpuOpts = tt::Options(tt::Device::cpu(), tt::DType::Int32).noGrad();
  auto slotGpu = tt::Tensor(slotData, {numTokens}, slotCpuOpts).to(tt::Device(tt::DeviceType::CUDA));

  auto posCpuOpts = tt::Options(tt::Device::cpu(), tt::DType::Int64).noGrad();
  auto posGpu = tt::Tensor(posData, {numTokens}, posCpuOpts).to(tt::Device(tt::DeviceType::CUDA));

  auto cacheOpts = tt::Options(tt::Device(tt::DeviceType::CUDA), tt::DType::BFloat16).noGrad();
  int64_t poolSize = static_cast<int64_t>(numBlocks) * numKvHeads * blockSize * headDim;
  tt::Tensor kCacheRef = tt::Tensor::zeros({poolSize}, cacheOpts);
  tt::Tensor vCacheRef = tt::Tensor::zeros({poolSize}, cacheOpts);
  tt::Tensor kCacheFused = tt::Tensor::zeros({poolSize}, cacheOpts);
  tt::Tensor vCacheFused = tt::Tensor::zeros({poolSize}, cacheOpts);

  // Reference: separate RMSNorm(K) + RoPE(K) + scatter
  // RMSNorm per-head: reshape to [numTokens*numKvHeads, headDim], normalize, reshape back
  auto keyFlat = keyGpu.view({static_cast<int64_t>(numTokens) * numKvHeads, headDim});
  auto keyNormed = tt::function::rmsNorm(keyFlat, {headDim}, normWeightGpu, eps);
  auto keyNormed3d = keyNormed.view({numTokens, numKvHeads, headDim});
  auto keyRotated = kernel::ropeApply(keyNormed3d, ropeCache, posGpu);
  kernel::scatterKVToCache(keyRotated, valGpu, kCacheRef, vCacheRef, slotGpu, blockSize);

  // Fused: normRopeScatterKVToCache
  kernel::normRopeScatterKVToCache(keyGpu, valGpu, kCacheFused, vCacheFused, slotGpu, blockSize, ropeCache, posGpu,
                                   normWeightGpu, eps);

  auto kRef = readFloats(kCacheRef);
  auto kFused = readFloats(kCacheFused);
  auto vRef = readFloats(vCacheRef);
  auto vFused = readFloats(vCacheFused);

  ASSERT_EQ(kRef.size(), kFused.size());
  ASSERT_EQ(vRef.size(), vFused.size());

  // BF16 accumulation in RMSNorm can have slight differences due to reduction order.
  // Use tolerance instead of exact match.
  for (size_t i = 0; i < kRef.size(); i++) {
    float diff = std::fabs(kRef[i] - kFused[i]);
    float threshold = 0.05f + 0.01f * std::fabs(kRef[i]);
    EXPECT_LE(diff, threshold) << "K cache mismatch at index " << i << ": ref=" << kRef[i] << " fused=" << kFused[i];
  }
  for (size_t i = 0; i < vRef.size(); i++) {
    EXPECT_EQ(vRef[i], vFused[i]) << "V cache mismatch at index " << i;
  }
}

std::string normRopeScatterTestName(const ::testing::TestParamInfo<RopeScatterParam>& info) {
  auto& p = info.param;
  return "T" + std::to_string(p.numTokens) + "_H" + std::to_string(p.numKvHeads) + "_D" +
         std::to_string(p.headDim) + "_BS" + std::to_string(p.blockSize);
}

INSTANTIATE_TEST_SUITE_P(configs, NormRopeScatterTest,
                         ::testing::Values(
                             RopeScatterParam{1, 8, 128, 16, 64},     // Qwen3 decode
                             RopeScatterParam{16, 8, 128, 16, 64},    // small prefill
                             RopeScatterParam{1, 32, 128, 16, 64},    // many heads
                             RopeScatterParam{4, 4, 64, 16, 32}       // smaller head dim
                         ),
                         normRopeScatterTestName);

// =============================================================================
// Test ropeApplyInplace matches ropeApply
// =============================================================================

struct RopeInplaceParam {
  int numTokens;
  int numHeads;
  int headDim;
};

class RopeInplaceTest : public ::testing::TestWithParam<RopeInplaceParam> {};

TEST_P(RopeInplaceTest, matches_out_of_place) {
  SKIP_IF_NO_CUDA();
  auto p = GetParam();
  const int maxPos = 4096;

  std::vector<float> inputData(p.numTokens * p.numHeads * p.headDim);
  fillRandom(inputData, 123, 0.5f);
  quantizeToBF16(inputData);

  std::vector<int64_t> posData(p.numTokens);
  for (int i = 0; i < p.numTokens; i++) posData[i] = i + 200;

  auto ropeCache = kernel::ropeInit(p.headDim, maxPos, 10000.f, nullptr,
                                    tt::Options(tt::Device(tt::DeviceType::CUDA), tt::DType::Float32).noGrad());

  auto posCpuOpts = tt::Options(tt::Device::cpu(), tt::DType::Int64).noGrad();
  auto posGpu = tt::Tensor(posData, {p.numTokens}, posCpuOpts).to(tt::Device(tt::DeviceType::CUDA));

  // Out-of-place reference
  auto inputRef = makeCudaTensor(inputData, {p.numTokens, p.numHeads, p.headDim}, tt::DType::BFloat16);
  auto outputRef = kernel::ropeApply(inputRef, ropeCache, posGpu);

  // In-place
  auto inputInplace = makeCudaTensor(inputData, {p.numTokens, p.numHeads, p.headDim}, tt::DType::BFloat16);
  kernel::ropeApplyInplace(inputInplace, ropeCache, posGpu);

  auto refVals = readFloats(outputRef);
  auto inplaceVals = readFloats(inputInplace);

  ASSERT_EQ(refVals.size(), inplaceVals.size());
  for (size_t i = 0; i < refVals.size(); i++) {
    EXPECT_EQ(refVals[i], inplaceVals[i]) << "Mismatch at index " << i;
  }
}

std::string ropeInplaceTestName(const ::testing::TestParamInfo<RopeInplaceParam>& info) {
  auto& p = info.param;
  return "T" + std::to_string(p.numTokens) + "_H" + std::to_string(p.numHeads) + "_D" + std::to_string(p.headDim);
}

INSTANTIATE_TEST_SUITE_P(configs, RopeInplaceTest,
                         ::testing::Values(
                             RopeInplaceParam{1, 16, 128},    // Qwen3 Q decode
                             RopeInplaceParam{32, 16, 128},   // prefill chunk
                             RopeInplaceParam{1, 32, 128},    // Llama-like
                             RopeInplaceParam{4, 8, 64}       // smaller config
                         ),
                         ropeInplaceTestName);

// =============================================================================
// Test embeddingLookup matches generic embedding
// =============================================================================

struct EmbedParam {
  int numTokens;
  int vocabSize;
  int hiddenSize;
};

class EmbeddingLookupTest : public ::testing::TestWithParam<EmbedParam> {};

TEST_P(EmbeddingLookupTest, matches_generic_embedding) {
  SKIP_IF_NO_CUDA();
  auto p = GetParam();

  // Create embedding table
  std::vector<float> tableData(p.vocabSize * p.hiddenSize);
  fillRandom(tableData, 42, 1.0f);
  quantizeToBF16(tableData);
  auto tableGpu = makeCudaTensor(tableData, {p.vocabSize, p.hiddenSize}, tt::DType::BFloat16);

  // Create token IDs (random valid indices)
  std::mt19937 rng(99);
  std::uniform_int_distribution<int64_t> dist(0, p.vocabSize - 1);
  std::vector<int64_t> tokenIds(p.numTokens);
  for (auto& id : tokenIds) id = dist(rng);

  auto idCpuOpts = tt::Options(tt::Device::cpu(), tt::DType::Int64).noGrad();
  auto idsGpu = tt::Tensor(tokenIds, {p.numTokens}, idCpuOpts).to(tt::Device(tt::DeviceType::CUDA));

  // Fast path
  auto fastResult = kernel::embeddingLookup(idsGpu, tableGpu);
  if (!fastResult.defined()) {
    GTEST_SKIP() << "embeddingLookup declined (numTokens=" << p.numTokens << " > 32)";
    return;
  }

  // Reference: manually extract rows
  auto fastVals = readFloats(fastResult);
  ASSERT_EQ(fastVals.size(), static_cast<size_t>(p.numTokens * p.hiddenSize));

  for (int t = 0; t < p.numTokens; t++) {
    int64_t tokenId = tokenIds[t];
    for (int d = 0; d < p.hiddenSize; d++) {
      float expected = tableData[tokenId * p.hiddenSize + d];
      float actual = fastVals[t * p.hiddenSize + d];
      EXPECT_EQ(expected, actual) << "Mismatch at token=" << t << " dim=" << d << " tokenId=" << tokenId;
    }
  }
}

std::string embedTestName(const ::testing::TestParamInfo<EmbedParam>& info) {
  auto& p = info.param;
  return "T" + std::to_string(p.numTokens) + "_V" + std::to_string(p.vocabSize) + "_H" + std::to_string(p.hiddenSize);
}

INSTANTIATE_TEST_SUITE_P(configs, EmbeddingLookupTest,
                         ::testing::Values(
                             EmbedParam{1, 151936, 1024},    // Qwen3 decode
                             EmbedParam{1, 32000, 4096},     // Llama 7B decode
                             EmbedParam{8, 151936, 1024},    // small batch
                             EmbedParam{32, 32000, 1024},    // max supported
                             EmbedParam{1, 1000, 256}        // small model
                         ),
                         embedTestName);

// =============================================================================
// Test embeddingLookup guard conditions
// =============================================================================

TEST(embedding_lookup_guard, returns_empty_for_large_batch) {
  SKIP_IF_NO_CUDA();
  auto opts = tt::Options(tt::Device(tt::DeviceType::CUDA), tt::DType::BFloat16).noGrad();
  auto idOpts = tt::Options(tt::Device(tt::DeviceType::CUDA), tt::DType::Int64).noGrad();
  tt::Tensor ids({64}, idOpts);   // > 32 tokens
  tt::Tensor table({1000, 256}, opts);
  auto result = kernel::embeddingLookup(ids, table);
  EXPECT_FALSE(result.defined());
}

TEST(embedding_lookup_guard, returns_empty_for_cpu) {
  SKIP_IF_NO_CUDA();
  auto cpuOpts = tt::Options(tt::Device::cpu(), tt::DType::BFloat16).noGrad();
  auto idOpts = tt::Options(tt::Device::cpu(), tt::DType::Int64).noGrad();
  tt::Tensor ids({1}, idOpts);
  tt::Tensor table({1000, 256}, cpuOpts);
  auto result = kernel::embeddingLookup(ids, table);
  EXPECT_FALSE(result.defined());
}

// =============================================================================
// Test padding tokens for fused kernels
// =============================================================================

TEST(rope_scatter_fused, padding_tokens_skipped) {
  SKIP_IF_NO_CUDA();

  const int numTokens = 4;
  const int numKvHeads = 8;
  const int headDim = 128;
  const int blockSize = 16;
  const int numBlocks = 8;
  const int maxPos = 4096;

  std::vector<float> keyData(numTokens * numKvHeads * headDim);
  std::vector<float> valData(numTokens * numKvHeads * headDim);
  fillRandom(keyData, 42, 0.5f);
  fillRandom(valData, 99, 0.5f);
  quantizeToBF16(keyData);
  quantizeToBF16(valData);

  std::vector<int32_t> slotData = {0, -1, 2, -1};
  std::vector<int64_t> posData = {100, 0, 102, 0};

  auto ropeCache = kernel::ropeInit(headDim, maxPos, 10000.f, nullptr,
                                    tt::Options(tt::Device(tt::DeviceType::CUDA), tt::DType::Float32).noGrad());

  auto keyGpu = makeCudaTensor(keyData, {numTokens, numKvHeads, headDim}, tt::DType::BFloat16);
  auto valGpu = makeCudaTensor(valData, {numTokens, numKvHeads, headDim}, tt::DType::BFloat16);

  auto slotCpuOpts = tt::Options(tt::Device::cpu(), tt::DType::Int32).noGrad();
  auto slotGpu = tt::Tensor(slotData, {numTokens}, slotCpuOpts).to(tt::Device(tt::DeviceType::CUDA));
  auto posCpuOpts = tt::Options(tt::Device::cpu(), tt::DType::Int64).noGrad();
  auto posGpu = tt::Tensor(posData, {numTokens}, posCpuOpts).to(tt::Device(tt::DeviceType::CUDA));

  auto cacheOpts = tt::Options(tt::Device(tt::DeviceType::CUDA), tt::DType::BFloat16).noGrad();
  int64_t poolSize = static_cast<int64_t>(numBlocks) * numKvHeads * blockSize * headDim;
  tt::Tensor kCache = tt::Tensor::zeros({poolSize}, cacheOpts);
  tt::Tensor vCache = tt::Tensor::zeros({poolSize}, cacheOpts);

  kernel::ropeScatterKVToCache(keyGpu, valGpu, kCache, vCache, slotGpu, blockSize, ropeCache, posGpu);

  auto kResult = readFloats(kCache);

  // Slot 0 should have been written
  bool slot0HasData = false;
  for (int i = 0; i < numKvHeads * headDim; i++) {
    if (kResult[i] != 0.f) { slot0HasData = true; break; }
  }
  EXPECT_TRUE(slot0HasData) << "Slot 0 should have been written";
}

TEST(norm_rope_scatter_fused, padding_tokens_skipped) {
  SKIP_IF_NO_CUDA();

  const int numTokens = 4;
  const int numKvHeads = 8;
  const int headDim = 128;
  const int blockSize = 16;
  const int numBlocks = 8;
  const int maxPos = 4096;
  const float eps = 1e-6f;

  std::vector<float> keyData(numTokens * numKvHeads * headDim);
  std::vector<float> valData(numTokens * numKvHeads * headDim);
  std::vector<float> normWeightData(headDim, 1.0f);
  fillRandom(keyData, 42, 0.5f);
  fillRandom(valData, 99, 0.5f);
  quantizeToBF16(keyData);
  quantizeToBF16(valData);

  std::vector<int32_t> slotData = {0, -1, 2, -1};
  std::vector<int64_t> posData = {100, 0, 102, 0};

  auto ropeCache = kernel::ropeInit(headDim, maxPos, 10000.f, nullptr,
                                    tt::Options(tt::Device(tt::DeviceType::CUDA), tt::DType::Float32).noGrad());

  auto keyGpu = makeCudaTensor(keyData, {numTokens, numKvHeads, headDim}, tt::DType::BFloat16);
  auto valGpu = makeCudaTensor(valData, {numTokens, numKvHeads, headDim}, tt::DType::BFloat16);
  auto normWeightGpu = makeCudaTensor(normWeightData, {headDim}, tt::DType::BFloat16);

  auto slotCpuOpts = tt::Options(tt::Device::cpu(), tt::DType::Int32).noGrad();
  auto slotGpu = tt::Tensor(slotData, {numTokens}, slotCpuOpts).to(tt::Device(tt::DeviceType::CUDA));
  auto posCpuOpts = tt::Options(tt::Device::cpu(), tt::DType::Int64).noGrad();
  auto posGpu = tt::Tensor(posData, {numTokens}, posCpuOpts).to(tt::Device(tt::DeviceType::CUDA));

  auto cacheOpts = tt::Options(tt::Device(tt::DeviceType::CUDA), tt::DType::BFloat16).noGrad();
  int64_t poolSize = static_cast<int64_t>(numBlocks) * numKvHeads * blockSize * headDim;
  tt::Tensor kCache = tt::Tensor::zeros({poolSize}, cacheOpts);
  tt::Tensor vCache = tt::Tensor::zeros({poolSize}, cacheOpts);

  kernel::normRopeScatterKVToCache(keyGpu, valGpu, kCache, vCache, slotGpu, blockSize, ropeCache, posGpu,
                                   normWeightGpu, eps);

  auto kResult = readFloats(kCache);

  // Slot 0 should have data
  bool slot0HasData = false;
  for (int i = 0; i < numKvHeads * headDim; i++) {
    if (kResult[i] != 0.f) { slot0HasData = true; break; }
  }
  EXPECT_TRUE(slot0HasData) << "Slot 0 should have been written";

  // Slot 1 (padding) area should be zero (never written)
  // slot=1 would map to blockId=0, blockOffset=1 in cache layout
  const int slot1Offset = 0 * numKvHeads * blockSize * headDim + 0 * blockSize * headDim + 1 * headDim;
  bool slot1IsZero = true;
  for (int i = 0; i < headDim; i++) {
    if (kResult[slot1Offset + i] != 0.f) { slot1IsZero = false; break; }
  }
  EXPECT_TRUE(slot1IsZero) << "Padding slot should not have been written";
}

}  // namespace tinygpt
