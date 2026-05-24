/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "Utils/CUDAUtils.h"
#include "kernel/GemvOps.h"
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

// Simulate BF16 precision on CPU: round float32 to bfloat16
inline float toBF16Precision(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  bits += 0x7FFF + ((bits >> 16) & 1);
  bits &= 0xFFFF0000u;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

// Simulate FP16 precision on CPU: round float32 to float16
// FP16: 1 sign + 5 exp + 10 mantissa (implicit 1 + 10 bits = 11 significant bits)
inline float toFP16Precision(float v) {
  // Use the standard conversion: float32 -> float16 -> float32
  // FP16 has 10 mantissa bits, so we mask off the lower 13 bits of float32 mantissa
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));

  int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFF) - 127;
  // Handle overflow/underflow for FP16 range
  if (exp > 15) return (bits & 0x80000000u) ? -65504.f : 65504.f;
  if (exp < -24) return 0.f;

  // Round mantissa: keep top 10 bits of the 23-bit mantissa
  // Round to nearest even: add bit 12 + sticky
  uint32_t mantissa = bits & 0x007FFFFFu;
  uint32_t roundBit = (mantissa >> 12) & 1;
  bits += (0x00000FFF + roundBit);
  bits &= 0xFFFFE000u;  // zero out lower 13 bits
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

// Reference CPU implementation: output[i] = dot(input, weight[i, :])
std::vector<float> gemvReference(const std::vector<float>& input, const std::vector<float>& weight, int N, int K) {
  std::vector<float> output(N, 0.f);
  for (int i = 0; i < N; i++) {
    float sum = 0.f;
    for (int j = 0; j < K; j++) {
      sum += input[j] * weight[i * K + j];
    }
    output[i] = sum;
  }
  return output;
}

// Create a CUDA tensor from float data at a given dtype
tt::Tensor makeCudaTensor(const std::vector<float>& data, const std::vector<int64_t>& shape, tt::DType dtype) {
  auto cpuOpts = tt::Options(tt::Device::cpu(), tt::DType::Float32).noGrad();
  tt::Tensor host(data, shape, cpuOpts);
  return host.to(tt::Device(tt::DeviceType::CUDA)).to(dtype);
}

// Read a CUDA tensor back to host floats
std::vector<float> readFloats(const tt::Tensor& tensor) {
  auto cpu = tensor.to(tt::DType::Float32).to(tt::Device::cpu());
  return cpu.toList<float>();
}

// Fill vector with random values in [-scale, scale]
void fillRandom(std::vector<float>& v, unsigned int seed, float scale = 1.f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-scale, scale);
  for (auto& x : v) {
    x = dist(rng);
  }
}

// Quantize to the precision of the given dtype
void quantizeToPrec(std::vector<float>& v, tt::DType dtype) {
  if (dtype == tt::DType::BFloat16) {
    for (auto& x : v) x = toBF16Precision(x);
  } else if (dtype == tt::DType::Float16) {
    for (auto& x : v) x = toFP16Precision(x);
  }
  // Float32: no quantization needed
}

// Get tolerance appropriate for the dtype
// BF16: ~7 bit mantissa → eps ≈ 2^-8 = 0.0039
// FP16: ~10 bit mantissa → eps ≈ 2^-11 = 0.00049
// FP32: ~23 bit mantissa → eps ≈ 2^-24
struct Tolerance {
  float rtol;
  float atol;
};

Tolerance getTolerance(tt::DType dtype) {
  switch (dtype) {
    case tt::DType::BFloat16: return {1e-2f, 5e-2f};
    case tt::DType::Float16:  return {5e-3f, 1e-2f};
    case tt::DType::Float32:  return {5e-5f, 5e-5f};
    default:                  return {1e-2f, 5e-2f};
  }
}

// Random input scale (FP16 range is limited to ±65504)
float getInputScale(tt::DType dtype) {
  return (dtype == tt::DType::Float16) ? 0.5f : 1.f;
}

// Compare with tolerance
void expectNear(const std::vector<float>& actual, const std::vector<float>& expected, Tolerance tol) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); i++) {
    float diff = std::fabs(actual[i] - expected[i]);
    float threshold = tol.atol + tol.rtol * std::fabs(expected[i]);
    EXPECT_LE(diff, threshold) << "Mismatch at index " << i << ": actual=" << actual[i]
                               << " expected=" << expected[i] << " diff=" << diff << " threshold=" << threshold;
  }
}

std::string dtypeName(tt::DType dtype) {
  switch (dtype) {
    case tt::DType::BFloat16: return "BF16";
    case tt::DType::Float16:  return "FP16";
    case tt::DType::Float32:  return "FP32";
    default:                  return "Unknown";
  }
}

}  // namespace

// =============================================================================
// Test gemvLmHead (always active, no N threshold)
// Parameterized by (N, K, DType)
// =============================================================================

struct GemvTestParam {
  int N;
  int K;
  tt::DType dtype;
};

class GemvLmHeadTest : public ::testing::TestWithParam<GemvTestParam> {};

TEST_P(GemvLmHeadTest, correctness_vs_cpu_reference) {
  SKIP_IF_NO_CUDA();
  auto param = GetParam();
  int N = param.N, K = param.K;
  tt::DType dtype = param.dtype;

  float scale = getInputScale(dtype);
  std::vector<float> inputHost(K);
  std::vector<float> weightHost(static_cast<size_t>(N) * K);
  fillRandom(inputHost, 42, scale);
  fillRandom(weightHost, 123, scale);

  // Quantize to target precision for fair comparison
  quantizeToPrec(inputHost, dtype);
  quantizeToPrec(weightHost, dtype);

  // CPU reference (in float32, on quantized values)
  auto expected = gemvReference(inputHost, weightHost, N, K);

  // GPU kernel
  auto inputGpu = makeCudaTensor(inputHost, {1, K}, dtype);
  auto weightGpu = makeCudaTensor(weightHost, {static_cast<int64_t>(N), static_cast<int64_t>(K)}, dtype);
  auto outputGpu = kernel::gemvLmHead(inputGpu, weightGpu);

  auto actual = readFloats(outputGpu);
  ASSERT_EQ(actual.size(), static_cast<size_t>(N));

  auto tol = getTolerance(dtype);
  expectNear(actual, expected, tol);
}

// Generate human-readable test names
std::string gemvTestName(const ::testing::TestParamInfo<GemvTestParam>& info) {
  auto& p = info.param;
  return "N" + std::to_string(p.N) + "_K" + std::to_string(p.K) + "_" + dtypeName(p.dtype);
}

INSTANTIATE_TEST_SUITE_P(gemv_lm_head, GemvLmHeadTest,
                         ::testing::Values(
                             // BF16 — typical LLM sizes
                             GemvTestParam{151936, 1024, tt::DType::BFloat16},  // lm_head Qwen3-0.6B
                             GemvTestParam{3072, 1024, tt::DType::BFloat16},    // qkv_proj
                             GemvTestParam{5632, 1024, tt::DType::BFloat16},    // gate_up_proj
                             GemvTestParam{1024, 1024, tt::DType::BFloat16},    // o_proj
                             GemvTestParam{1024, 2816, tt::DType::BFloat16},    // down_proj
                             GemvTestParam{8, 256, tt::DType::BFloat16},        // minimal
                             GemvTestParam{32000, 4096, tt::DType::BFloat16},   // Llama 7B

                             // FP16 — same sizes
                             GemvTestParam{151936, 1024, tt::DType::Float16},
                             GemvTestParam{3072, 1024, tt::DType::Float16},
                             GemvTestParam{5632, 1024, tt::DType::Float16},
                             GemvTestParam{1024, 1024, tt::DType::Float16},
                             GemvTestParam{1024, 2816, tt::DType::Float16},
                             GemvTestParam{32000, 4096, tt::DType::Float16},

                             // FP32
                             GemvTestParam{3072, 1024, tt::DType::Float32},
                             GemvTestParam{1024, 1024, tt::DType::Float32},
                             GemvTestParam{4096, 2048, tt::DType::Float32}
                         ),
                         gemvTestName);

// =============================================================================
// Test gemvLinear (with N threshold check)
// =============================================================================

class GemvLinearTest : public ::testing::TestWithParam<GemvTestParam> {};

TEST_P(GemvLinearTest, correctness_vs_cpu_reference) {
  SKIP_IF_NO_CUDA();
  auto param = GetParam();
  int N = param.N, K = param.K;
  tt::DType dtype = param.dtype;

  float scale = getInputScale(dtype);
  std::vector<float> inputHost(K);
  std::vector<float> weightHost(static_cast<size_t>(N) * K);
  fillRandom(inputHost, 77, scale);
  fillRandom(weightHost, 999, scale);

  quantizeToPrec(inputHost, dtype);
  quantizeToPrec(weightHost, dtype);

  auto expected = gemvReference(inputHost, weightHost, N, K);

  auto inputGpu = makeCudaTensor(inputHost, {1, K}, dtype);
  auto weightGpu = makeCudaTensor(weightHost, {static_cast<int64_t>(N), static_cast<int64_t>(K)}, dtype);
  auto outputGpu = kernel::gemvLinear(inputGpu, weightGpu);

  if (!outputGpu.defined()) {
    GTEST_SKIP() << "N=" << N << " below device threshold, gemvLinear returned empty (expected)";
    return;
  }

  auto actual = readFloats(outputGpu);
  ASSERT_EQ(actual.size(), static_cast<size_t>(N));

  auto tol = getTolerance(dtype);
  expectNear(actual, expected, tol);
}

INSTANTIATE_TEST_SUITE_P(gemv_linear, GemvLinearTest,
                         ::testing::Values(
                             // BF16
                             GemvTestParam{3072, 1024, tt::DType::BFloat16},
                             GemvTestParam{5632, 1024, tt::DType::BFloat16},
                             GemvTestParam{1024, 1024, tt::DType::BFloat16},    // likely below threshold
                             GemvTestParam{8192, 1024, tt::DType::BFloat16},
                             // FP16
                             GemvTestParam{3072, 1024, tt::DType::Float16},
                             GemvTestParam{5632, 1024, tt::DType::Float16},
                             GemvTestParam{1024, 1024, tt::DType::Float16},
                             GemvTestParam{8192, 1024, tt::DType::Float16},
                             // FP32
                             GemvTestParam{3072, 1024, tt::DType::Float32},
                             GemvTestParam{4096, 4096, tt::DType::Float32}
                         ),
                         gemvTestName);

// =============================================================================
// Test gemvLinear returns empty for invalid/unsupported inputs
// =============================================================================

TEST(gemv_linear_guard, returns_empty_for_batch_gt1) {
  SKIP_IF_NO_CUDA();
  auto opts = tt::Options(tt::Device(tt::DeviceType::CUDA), tt::DType::BFloat16).noGrad();
  tt::Tensor input({2, 1024}, opts);  // M=2, not 1
  tt::Tensor weight({4096, 1024}, opts);
  auto result = kernel::gemvLinear(input, weight);
  EXPECT_FALSE(result.defined());
}

TEST(gemv_linear_guard, returns_empty_for_mismatched_k) {
  SKIP_IF_NO_CUDA();
  auto opts = tt::Options(tt::Device(tt::DeviceType::CUDA), tt::DType::BFloat16).noGrad();
  tt::Tensor input({1, 1024}, opts);
  tt::Tensor weight({4096, 512}, opts);  // K mismatch
  auto result = kernel::gemvLinear(input, weight);
  EXPECT_FALSE(result.defined());
}

TEST(gemv_linear_guard, returns_empty_for_unaligned_k) {
  SKIP_IF_NO_CUDA();
  auto opts = tt::Options(tt::Device(tt::DeviceType::CUDA), tt::DType::Float16).noGrad();
  tt::Tensor input({1, 100}, opts);  // K=100, not divisible by 8
  tt::Tensor weight({4096, 100}, opts);
  auto result = kernel::gemvLinear(input, weight);
  EXPECT_FALSE(result.defined());
}

// =============================================================================
// Precision tests with known patterns
// =============================================================================

class GemvPrecisionTest : public ::testing::TestWithParam<tt::DType> {};

TEST_P(GemvPrecisionTest, identity_like_pattern) {
  SKIP_IF_NO_CUDA();
  tt::DType dtype = GetParam();

  // input = [1, 0, 0, ..., 0] → output should equal weight[:, 0]
  const int N = 2048;
  const int K = 1024;

  std::vector<float> inputHost(K, 0.f);
  inputHost[0] = 1.f;
  std::vector<float> weightHost(static_cast<size_t>(N) * K);
  float scale = getInputScale(dtype);
  fillRandom(weightHost, 42, scale);
  quantizeToPrec(weightHost, dtype);

  std::vector<float> expected(N);
  for (int i = 0; i < N; i++) {
    expected[i] = weightHost[i * K + 0];
  }

  auto inputGpu = makeCudaTensor(inputHost, {1, K}, dtype);
  auto weightGpu = makeCudaTensor(weightHost, {N, static_cast<int64_t>(K)}, dtype);
  auto outputGpu = kernel::gemvLmHead(inputGpu, weightGpu);
  auto actual = readFloats(outputGpu);

  // Single non-zero input means no accumulation error — result should be exact
  Tolerance tol = {1e-5f, 1e-5f};
  expectNear(actual, expected, tol);
}

TEST_P(GemvPrecisionTest, all_ones_vector) {
  SKIP_IF_NO_CUDA();
  tt::DType dtype = GetParam();

  const int N = 4096;
  const int K = 1024;

  std::vector<float> inputHost(K, 1.0f);
  std::vector<float> weightHost(static_cast<size_t>(N) * K);
  for (int i = 0; i < N; i++) {
    float val = static_cast<float>(i + 1) * 0.001f;
    std::fill(weightHost.begin() + i * K, weightHost.begin() + (i + 1) * K, val);
  }
  quantizeToPrec(inputHost, dtype);
  quantizeToPrec(weightHost, dtype);

  auto expected = gemvReference(inputHost, weightHost, N, K);

  auto inputGpu = makeCudaTensor(inputHost, {1, K}, dtype);
  auto weightGpu = makeCudaTensor(weightHost, {N, static_cast<int64_t>(K)}, dtype);
  auto outputGpu = kernel::gemvLmHead(inputGpu, weightGpu);
  auto actual = readFloats(outputGpu);

  auto tol = getTolerance(dtype);
  // accumulated sums have higher error due to tile reduction order
  if (dtype == tt::DType::Float32) {
    tol.rtol *= 4.f;
    tol.atol *= 4.f;
  } else {
    tol.atol *= 2.f;
  }
  expectNear(actual, expected, tol);
}

std::string precTestName(const ::testing::TestParamInfo<tt::DType>& info) {
  return dtypeName(info.param);
}

INSTANTIATE_TEST_SUITE_P(gemv_precision, GemvPrecisionTest,
                         ::testing::Values(tt::DType::BFloat16, tt::DType::Float16, tt::DType::Float32),
                         precTestName);

// =============================================================================
// Test consistency between gemvLmHead and gemvLinear (when both active)
// =============================================================================

class GemvConsistencyTest : public ::testing::TestWithParam<tt::DType> {};

TEST_P(GemvConsistencyTest, lm_head_and_linear_produce_same_result) {
  SKIP_IF_NO_CUDA();
  tt::DType dtype = GetParam();
  const int N = 4096;
  const int K = 1024;

  float scale = getInputScale(dtype);
  std::vector<float> inputHost(K);
  std::vector<float> weightHost(static_cast<size_t>(N) * K);
  fillRandom(inputHost, 2024, scale);
  fillRandom(weightHost, 2025, scale);
  quantizeToPrec(inputHost, dtype);
  quantizeToPrec(weightHost, dtype);

  auto inputGpu = makeCudaTensor(inputHost, {1, K}, dtype);
  auto weightGpu = makeCudaTensor(weightHost, {N, static_cast<int64_t>(K)}, dtype);

  auto out1 = kernel::gemvLmHead(inputGpu, weightGpu);
  auto out2 = kernel::gemvLinear(inputGpu, weightGpu);

  if (!out2.defined()) {
    GTEST_SKIP() << "N=" << N << " below threshold on this device";
    return;
  }

  auto v1 = readFloats(out1);
  auto v2 = readFloats(out2);
  ASSERT_EQ(v1.size(), v2.size());

  // Should be bit-identical since they call the same kernel
  for (size_t i = 0; i < v1.size(); i++) {
    EXPECT_EQ(v1[i], v2[i]) << "Mismatch at index " << i << " (dtype=" << dtypeName(dtype) << ")";
  }
}

INSTANTIATE_TEST_SUITE_P(gemv_consistency, GemvConsistencyTest,
                         ::testing::Values(tt::DType::BFloat16, tt::DType::Float16, tt::DType::Float32),
                         precTestName);

}  // namespace tinygpt
