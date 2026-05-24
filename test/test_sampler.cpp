/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <unordered_set>
#include <vector>

#include "Functions.h"
#include "Utils/CUDAUtils.h"
#include "Utils/RandomGenerator.h"
#include "engine/Sampler.h"
#include "kernel/SamplerOps.h"
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

// Build a [batch, vocab] fp32 CUDA logits tensor from a 2D host vector.
tt::Tensor makeLogits(const std::vector<std::vector<float>>& rows) {
  const auto batch = static_cast<int64_t>(rows.size());
  const auto vocab = static_cast<int64_t>(rows[0].size());
  std::vector<float> flat;
  flat.reserve(static_cast<size_t>(batch * vocab));
  for (const auto& row : rows) {
    EXPECT_EQ(static_cast<int64_t>(row.size()), vocab);
    flat.insert(flat.end(), row.begin(), row.end());
  }
  // Create as a CPU fp32 tensor first, then move to CUDA. This mirrors the
  // way inference code ingests logits via Tensor::to(device).
  auto cpuOpts = tt::Options(tt::Device::cpu(), tt::DType::Float32).noGrad();
  tt::Tensor host(flat, {batch, vocab}, cpuOpts);
  return host.to(tt::Device(tt::DeviceType::CUDA));
}

tt::Tensor makeLogits(const std::vector<float>& singleRow) {
  return makeLogits(std::vector<std::vector<float>>{singleRow});
}

// Read the int64 sampled tokens out as a host vector.
std::vector<int64_t> readTokens(const tt::Tensor& sampled) {
  EXPECT_EQ(sampled.dim(), 2);
  EXPECT_EQ(sampled.size(-1), 1);
  auto cpu = sampled.to(tt::Device::cpu());
  return cpu.toList<int64_t>();
}

// Drive the kernel directly (bypassing the Sampler class) for tight unit
// tests. Uses a deterministic globalSeq derived from the test state.
int64_t sampleOnceDirect(const tt::Tensor& logits, const kernel::SamplingParams& params, uint64_t seed, uint64_t seq) {
  auto out = kernel::fusedSample(logits, params, seed, seq);
  return readTokens(out).front();
}

}  // namespace

// -----------------------------------------------------------------------------
// 1. Greedy fast-path
// -----------------------------------------------------------------------------
TEST(sampler, greedy_temperature_zero) {
  SKIP_IF_NO_CUDA();

  //   logits: [0.1, 5.0, 2.3, -1.0, 3.7]  -> argmax = index 1
  auto logits = makeLogits({0.1f, 5.0f, 2.3f, -1.0f, 3.7f});
  auto params = kernel::SamplingParams::greedy();

  // Seed does not matter for greedy — run a few times to prove determinism.
  for (uint64_t seq = 0; seq < 4; ++seq) {
    const auto tok = sampleOnceDirect(logits, params, /*seed=*/42, /*seq=*/seq);
    EXPECT_EQ(tok, 1);
  }
}

TEST(sampler, greedy_ties_prefer_lowest_index) {
  SKIP_IF_NO_CUDA();

  // Two tokens tied for the max — determinism requires we always pick the
  // smaller index (matches the ArgMaxOp tie-break rule).
  auto logits = makeLogits({3.0f, 5.0f, 5.0f, 1.0f});
  auto params = kernel::SamplingParams::greedy();
  EXPECT_EQ(sampleOnceDirect(logits, params, 0, 0), 1);
}

// -----------------------------------------------------------------------------
// 2. Temperature-only Gumbel-Max path — empirical distribution check
// -----------------------------------------------------------------------------
TEST(sampler, temperature_only_distribution) {
  SKIP_IF_NO_CUDA();

  auto logits = makeLogits({3.f, 1.f, -1.f});

  kernel::SamplingParams params;
  params.temperature = 1.f;
  params.topK = 3;   // == vocab_size, mathematically equivalent to no filter
  params.topP = 1.f;
  params.minP = 0.f;

  constexpr int N = 4096;
  std::vector<int> hist(3, 0);
  for (int i = 0; i < N; ++i) {
    const auto tok = sampleOnceDirect(logits, params, /*seed=*/static_cast<uint64_t>(i * 7 + 31), /*seq=*/static_cast<uint64_t>(i));
    ASSERT_GE(tok, 0);
    ASSERT_LT(tok, 3);
    hist[tok]++;
  }

  // softmax([3, 1, -1]) ~ [0.867, 0.117, 0.016]
  const float p0 = static_cast<float>(hist[0]) / N;
  const float p1 = static_cast<float>(hist[1]) / N;
  const float p2 = static_cast<float>(hist[2]) / N;
  EXPECT_NEAR(p0, 0.867f, 0.08f);
  EXPECT_NEAR(p1, 0.117f, 0.08f);
  EXPECT_NEAR(p2, 0.016f, 0.04f);
}

TEST(sampler, high_temperature_flattens_distribution) {
  SKIP_IF_NO_CUDA();

  auto logits = makeLogits({3.f, 1.f, -1.f});

  // T=100 makes the distribution almost uniform
  kernel::SamplingParams params;
  params.temperature = 100.f;
  params.topK = 3;  // == vocab_size

  constexpr int N = 6000;
  std::vector<int> hist(3, 0);
  for (int i = 0; i < N; ++i) {
    const auto tok = sampleOnceDirect(logits, params, /*seed=*/static_cast<uint64_t>(i * 13 + 5), /*seq=*/static_cast<uint64_t>(i));
    hist[tok]++;
  }

  // each bin should be roughly N/3
  for (int v : hist) {
    EXPECT_GE(v, N / 6);
    EXPECT_LE(v, N * 2 / 3);
  }
}

// -----------------------------------------------------------------------------
// 3. top-k filtering
// -----------------------------------------------------------------------------
TEST(sampler, topk_excludes_non_topk_tokens) {
  SKIP_IF_NO_CUDA();

  //   logits = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
  // with top-k = 3, only indices {0, 1, 2} may be sampled.
  std::vector<float> row(10);
  for (int i = 0; i < 10; ++i) row[i] = static_cast<float>(10 - i);
  auto logits = makeLogits(row);

  kernel::SamplingParams params;
  params.temperature = 1.f;
  params.topK = 3;

  std::unordered_set<int64_t> seen;
  for (int i = 0; i < 500; ++i) {
    const auto tok = sampleOnceDirect(logits, params, /*seed=*/1, /*seq=*/static_cast<uint64_t>(i));
    seen.insert(tok);
  }

  for (auto t : seen) {
    EXPECT_GE(t, 0);
    EXPECT_LT(t, 3) << "top-k leaked token " << t;
  }
  // We should actually hit all three — otherwise our mass estimate is off.
  EXPECT_EQ(seen.size(), 3u);
}

TEST(sampler, topk_one_equals_greedy) {
  SKIP_IF_NO_CUDA();

  auto logits = makeLogits({0.f, 5.f, 2.f, -1.f, 3.f});
  kernel::SamplingParams params;
  params.temperature = 1.f;
  params.topK = 1;

  for (uint64_t seq = 0; seq < 8; ++seq) {
    EXPECT_EQ(sampleOnceDirect(logits, params, /*seed=*/99, /*seq=*/seq), 1);
  }
}

TEST(sampler, topk_larger_than_vocab_is_safe) {
  SKIP_IF_NO_CUDA();

  // Kernel must clamp topK to vocab internally — otherwise it would either
  // mis-rank or assert. Test with small vocab + topK way larger.
  auto logits = makeLogits({1.f, 2.f, 3.f});
  kernel::SamplingParams params;
  params.temperature = 1.f;
  params.topK = 10000;

  for (uint64_t seq = 0; seq < 16; ++seq) {
    const auto tok = sampleOnceDirect(logits, params, /*seed=*/5, /*seq=*/seq);
    EXPECT_GE(tok, 0);
    EXPECT_LT(tok, 3);
  }
}

// -----------------------------------------------------------------------------
// 4. top-p filtering
// -----------------------------------------------------------------------------
TEST(sampler, topp_excludes_low_mass_tail) {
  SKIP_IF_NO_CUDA();

  //   softmax([6, 5, -5, -5, -5, -5]) ~ [0.731, 0.269, tiny, tiny, tiny, tiny]
  // With top-p = 0.9, the first two tokens carry >0.9 of the mass together,
  // so only {0, 1} should ever be sampled.
  auto logits = makeLogits({6.f, 5.f, -5.f, -5.f, -5.f, -5.f});

  kernel::SamplingParams params;
  params.temperature = 1.f;
  params.topP = 0.9f;

  std::unordered_set<int64_t> seen;
  for (int i = 0; i < 1000; ++i) {
    const auto tok = sampleOnceDirect(logits, params, /*seed=*/1, /*seq=*/static_cast<uint64_t>(i));
    seen.insert(tok);
  }

  for (auto t : seen) {
    EXPECT_LT(t, 2) << "top-p leaked tail token " << t;
  }
}

TEST(sampler, topp_always_keeps_argmax) {
  SKIP_IF_NO_CUDA();

  // Degenerate top-p = 0.01: the first candidate's prob already exceeds p,
  // so the kernel must still return it (never returns "empty set").
  auto logits = makeLogits({6.f, 5.f, 4.f});

  kernel::SamplingParams params;
  params.temperature = 1.f;
  params.topP = 0.01f;

  for (uint64_t seq = 0; seq < 8; ++seq) {
    EXPECT_EQ(sampleOnceDirect(logits, params, /*seed=*/1, /*seq=*/seq), 0);
  }
}

// -----------------------------------------------------------------------------
// 5. min-p filtering
// -----------------------------------------------------------------------------
TEST(sampler, minp_excludes_below_relative_threshold) {
  SKIP_IF_NO_CUDA();

  //   softmax([6, 5.9, 1]) ~ [0.510, 0.461, 0.029]
  // max_prob = 0.510, min_p = 0.5 -> threshold = 0.255, kept: tokens 0, 1.
  auto logits = makeLogits({6.f, 5.9f, 1.f});

  kernel::SamplingParams params;
  params.temperature = 1.f;
  params.minP = 0.5f;

  std::unordered_set<int64_t> seen;
  for (int i = 0; i < 800; ++i) {
    const auto tok = sampleOnceDirect(logits, params, /*seed=*/2, /*seq=*/static_cast<uint64_t>(i));
    seen.insert(tok);
  }
  for (auto t : seen) {
    EXPECT_LT(t, 2) << "min-p leaked tail token " << t;
  }
}

// -----------------------------------------------------------------------------
// 6. Per-request seed reproducibility
// -----------------------------------------------------------------------------
TEST(sampler, per_request_seed_reproducible) {
  SKIP_IF_NO_CUDA();

  // flat distribution so different seeds actually diverge
  auto logits = makeLogits({0.1f, 0.3f, 0.2f, 0.4f, 0.0f, 0.5f, 0.25f});

  kernel::SamplingParams p1;
  p1.temperature = 1.0f;
  p1.topK = 7;  // == vocab_size
  p1.seed = 12345;

  kernel::SamplingParams p2 = p1;
  p2.seed = 67890;

  // same seed + same globalSeq => same output (determinism)
  for (uint64_t seq = 0; seq < 10; ++seq) {
    const auto a = sampleOnceDirect(logits, p1, 0, seq);
    const auto b = sampleOnceDirect(logits, p1, 999, seq);
    EXPECT_EQ(a, b) << "per-request seed should override global seed";
  }

  // different per-request seeds should produce different draws somewhere
  bool diverged = false;
  for (uint64_t seq = 0; seq < 100; ++seq) {
    const auto a = sampleOnceDirect(logits, p1, 0, seq);
    const auto b = sampleOnceDirect(logits, p2, 0, seq);
    if (a != b) {
      diverged = true;
      break;
    }
  }
  EXPECT_TRUE(diverged);
}

// -----------------------------------------------------------------------------
// 7. Batched sampling — mixed greedy + stochastic in a single launch
// -----------------------------------------------------------------------------
TEST(sampler, batched_mixed_configs) {
  SKIP_IF_NO_CUDA();

  //   Row 0: strongly peaked at index 2 — greedy must pick 2.
  //   Row 1: peaked at index 0 — top-k=1 must pick 0.
  //   Row 2: flat(ish) — top-k=2 restricts to {1, 2}.
  auto logits = makeLogits({
      {0.1f, 0.2f, 9.0f, 0.5f},
      {7.0f, 2.0f, 1.0f, 0.5f},
      {0.0f, 2.0f, 3.0f, 0.1f},
  });

  std::vector<kernel::SamplingParams> perRow(3);
  perRow[0].temperature = 0.f;  // greedy
  perRow[1].temperature = 1.f;
  perRow[1].topK = 1;
  perRow[2].temperature = 1.f;
  perRow[2].topK = 2;  // must pick from {1, 2}

  // Run many times to cover stochastic assertions on rows 1 and 2.
  for (uint64_t seq = 0; seq < 100; ++seq) {
    auto out = kernel::fusedSample(logits, perRow.data(), 3, /*globalSeed=*/11, /*globalSeq=*/seq);
    auto toks = readTokens(out);
    ASSERT_EQ(toks.size(), 3u);
    EXPECT_EQ(toks[0], 2);  // greedy determinism
    EXPECT_EQ(toks[1], 0);  // top-k=1 determinism
    EXPECT_GE(toks[2], 1);
    EXPECT_LE(toks[2], 2);
  }
}

// -----------------------------------------------------------------------------
// 8. Dtype parity (greedy path is sensitive to precision of argmax)
// -----------------------------------------------------------------------------
TEST(sampler, greedy_parity_across_dtypes) {
  SKIP_IF_NO_CUDA();

  std::vector<float> row = {0.1f, 5.0f, 2.3f, -1.0f, 3.7f};
  auto cpuOpts = tt::Options(tt::Device::cpu(), tt::DType::Float32).noGrad();
  tt::Tensor host(row, {1, static_cast<int64_t>(row.size())}, cpuOpts);

  auto cuda = tt::Device(tt::DeviceType::CUDA);
  auto fp32Cuda = host.to(cuda);
  // Cast on device where bf16 / fp16 casts are well-supported.
  auto bf16Cuda = fp32Cuda.to(tt::DType::BFloat16);
  auto fp16Cuda = fp32Cuda.to(tt::DType::Float16);

  auto params = kernel::SamplingParams::greedy();
  EXPECT_EQ(sampleOnceDirect(fp32Cuda, params, 0, 0), 1);
  EXPECT_EQ(sampleOnceDirect(bf16Cuda, params, 0, 0), 1);
  EXPECT_EQ(sampleOnceDirect(fp16Cuda, params, 0, 0), 1);
}

// -----------------------------------------------------------------------------
// 9. Small-vocab model safety (vocab < kDefaultImplicitTopK)
// -----------------------------------------------------------------------------
TEST(sampler, small_vocab_with_implicit_topk) {
  SKIP_IF_NO_CUDA();

  // vocab = 5 is much smaller than kDefaultImplicitTopK (256). The kernel
  // should clamp its internal candidate budget to vocab and still produce
  // valid samples when the user asks for top-p only.
  auto logits = makeLogits({0.5f, 0.7f, 0.6f, 0.4f, 0.3f});

  kernel::SamplingParams params;
  params.temperature = 1.f;
  params.topP = 0.9f;  // no explicit topK — triggers the implicit path

  for (uint64_t seq = 0; seq < 32; ++seq) {
    const auto tok = sampleOnceDirect(logits, params, /*seed=*/3, /*seq=*/seq);
    EXPECT_GE(tok, 0);
    EXPECT_LT(tok, 5);
  }
}

// -----------------------------------------------------------------------------
// 10. High-level Sampler config end-to-end
// -----------------------------------------------------------------------------
TEST(sampler, high_level_greedy) {
  SKIP_IF_NO_CUDA();

  auto logits = makeLogits({0.1f, 5.0f, 2.3f, -1.0f, 3.7f});
  SamplerConfig cfg;  // default: temperature=0 => greedy
  Sampler sampler(cfg);
  EXPECT_FALSE(sampler.doSample());

  auto out = tt::function::argmax(logits, -1, true);
  EXPECT_EQ(readTokens(out).front(), 1);
}

TEST(sampler, high_level_sampling_routes_to_fused_kernel) {
  SKIP_IF_NO_CUDA();

  auto logits = makeLogits({0.1f, 5.0f, 2.3f, -1.0f, 3.7f});

  SamplerConfig cfg;
  cfg.temperature = 1.0f;
  cfg.topK = 3;
  cfg.topP = 0.9f;
  cfg.minP = 0.01f;
  cfg.seed = 42;

  Sampler sampler(cfg);
  EXPECT_TRUE(sampler.doSample());

  // top-k=3 on this distribution allows {1, 4, 2} — assert we never escape.
  std::unordered_set<int64_t> seen;
  for (int i = 0; i < 100; ++i) {
    const auto globalSeed = tt::RandomGeneratorCUDA::getSeed();
    const auto globalSeq = tt::RandomGeneratorCUDA::nextSequence();
    auto out = kernel::fusedSample(logits, sampler.params(), globalSeed, globalSeq);
    seen.insert(readTokens(out).front());
  }
  const std::unordered_set<int64_t> allowed = {1, 2, 4};
  for (auto t : seen) {
    EXPECT_TRUE(allowed.count(t) == 1) << "unexpected token " << t;
  }
}

TEST(sampler, high_level_config_temperature_zero_is_greedy) {
  SKIP_IF_NO_CUDA();

  // HuggingFace convention: temperature <= 0 means greedy regardless of
  // other knobs. Sampler::doSample() must reflect this.
  SamplerConfig cfg;
  cfg.temperature = 0.f;
  cfg.topK = 50;
  cfg.topP = 0.9f;
  Sampler sampler(cfg);
  EXPECT_FALSE(sampler.doSample());
}

// -----------------------------------------------------------------------------
// 11. Combined filters: top-k + top-p
// -----------------------------------------------------------------------------
TEST(sampler, topk_plus_topp_combined) {
  SKIP_IF_NO_CUDA();

  //   logits = [10, 9, 8, 7, 6, 5] -> top 4 by logit: {0, 1, 2, 3}
  //   softmax over top-4 ([10, 9, 8, 7]) ~ [0.64, 0.24, 0.088, 0.032]
  //   top-p = 0.8: cumsum reaches 0.8 at index 1 (0.64+0.24=0.88 >= 0.8)
  //   -> only {0, 1} should ever be sampled.
  auto logits = makeLogits({10.f, 9.f, 8.f, 7.f, 6.f, 5.f});

  kernel::SamplingParams params;
  params.temperature = 1.f;
  params.topK = 4;
  params.topP = 0.8f;

  std::unordered_set<int64_t> seen;
  for (int i = 0; i < 500; ++i) {
    const auto tok = sampleOnceDirect(logits, params, /*seed=*/1, /*seq=*/static_cast<uint64_t>(i));
    seen.insert(tok);
  }

  for (auto t : seen) {
    EXPECT_LT(t, 2) << "top-k + top-p leaked token " << t;
  }
  // Both tokens should appear (significant mass on each).
  EXPECT_GE(seen.size(), 2u);
}

// -----------------------------------------------------------------------------
// 12. Combined filters: top-k + min-p
// -----------------------------------------------------------------------------
TEST(sampler, topk_plus_minp_combined) {
  SKIP_IF_NO_CUDA();

  //   logits = [6, 5.9, 1, 0.5] -> top 3 by logit: {0, 1, 2}
  //   softmax over top-3 ([6, 5.9, 1]) ~ [0.505, 0.456, 0.034]
  //   min-p = 0.5: threshold = 0.505 * 0.5 = 0.253, kept: {0, 1}
  auto logits = makeLogits({6.f, 5.9f, 1.f, 0.5f});

  kernel::SamplingParams params;
  params.temperature = 1.f;
  params.topK = 3;
  params.minP = 0.5f;

  std::unordered_set<int64_t> seen;
  for (int i = 0; i < 500; ++i) {
    const auto tok = sampleOnceDirect(logits, params, /*seed=*/3, /*seq=*/static_cast<uint64_t>(i));
    seen.insert(tok);
  }

  for (auto t : seen) {
    EXPECT_LT(t, 2) << "top-k + min-p leaked token " << t;
  }
  EXPECT_GE(seen.size(), 2u);
}

// -----------------------------------------------------------------------------
// 13. Combined filters: top-k + top-p + min-p all active
// -----------------------------------------------------------------------------
TEST(sampler, topk_topp_minp_all_active) {
  SKIP_IF_NO_CUDA();

  //   logits = [10, 9.9, 8, 5, 4, 3, 2, 1]
  //   top-k = 5 keeps {0,1,2,3,4}
  //   softmax([10,9.9,8,5,4]) ~ [0.476, 0.432, 0.065, 0.003, 0.001]
  //   top-p = 0.95: cum = 0.476 -> 0.908 -> 0.973 >= 0.95 at i=2; kept: {0,1,2}
  //   min-p = 0.1: threshold = 0.476*0.1 = 0.048; 0.065 >= 0.048 so {0,1,2}
  //   Survivors: {0, 1, 2}
  auto logits = makeLogits({10.f, 9.9f, 8.f, 5.f, 4.f, 3.f, 2.f, 1.f});

  kernel::SamplingParams params;
  params.temperature = 1.f;
  params.topK = 5;
  params.topP = 0.95f;
  params.minP = 0.1f;

  std::unordered_set<int64_t> seen;
  for (int i = 0; i < 500; ++i) {
    const auto tok = sampleOnceDirect(logits, params, /*seed=*/7, /*seq=*/static_cast<uint64_t>(i));
    seen.insert(tok);
  }

  for (auto t : seen) {
    EXPECT_LE(t, 2) << "all-three-filters leaked token " << t;
  }
  EXPECT_GE(seen.size(), 2u);
}

// -----------------------------------------------------------------------------
// 14. Edge case: single-token vocab
// -----------------------------------------------------------------------------
TEST(sampler, vocab_one_is_safe) {
  SKIP_IF_NO_CUDA();

  // Vocab = 1: the only legal output is token 0, regardless of sampling knobs.
  auto logits = makeLogits(std::vector<float>{42.f});

  // Greedy path.
  EXPECT_EQ(sampleOnceDirect(logits, kernel::SamplingParams::greedy(), 0, 0), 0);

  // Temperature-only path.
  kernel::SamplingParams tempOnly;
  tempOnly.temperature = 1.5f;
  EXPECT_EQ(sampleOnceDirect(logits, tempOnly, 0, 0), 0);

  // top-k path (topK=1, clamped to vocab=1).
  kernel::SamplingParams topkP;
  topkP.temperature = 1.f;
  topkP.topK = 5;  // larger than vocab — must be clamped
  EXPECT_EQ(sampleOnceDirect(logits, topkP, 0, 0), 0);

  // top-p path.
  kernel::SamplingParams toppP;
  toppP.temperature = 1.f;
  toppP.topP = 0.5f;
  EXPECT_EQ(sampleOnceDirect(logits, toppP, 0, 0), 0);
}

// -----------------------------------------------------------------------------
// 15. Edge case: all logits identical — distribution should be uniform
// -----------------------------------------------------------------------------
TEST(sampler, identical_logits_uniform) {
  SKIP_IF_NO_CUDA();

  // five identical logits: softmax is exactly uniform (20% each)
  auto logits = makeLogits({1.f, 1.f, 1.f, 1.f, 1.f});

  kernel::SamplingParams params;
  params.temperature = 1.f;
  params.topK = 5;  // == vocab_size

  constexpr int N = 5000;
  std::vector<int> hist(5, 0);
  for (int i = 0; i < N; ++i) {
    const auto tok = sampleOnceDirect(logits, params, /*seed=*/static_cast<uint64_t>(i * 11 + 3), /*seq=*/static_cast<uint64_t>(i));
    ASSERT_GE(tok, 0);
    ASSERT_LT(tok, 5);
    hist[tok]++;
  }

  for (int v : hist) {
    EXPECT_GE(v, N / 12);
    EXPECT_LE(v, N * 2 / 5);
  }
}

// -----------------------------------------------------------------------------
// 16. Edge case: negative temperature is greedy
// -----------------------------------------------------------------------------
TEST(sampler, negative_temperature_is_greedy) {
  SKIP_IF_NO_CUDA();

  auto logits = makeLogits({0.1f, 5.0f, 2.3f, -1.0f, 3.7f});

  kernel::SamplingParams params;
  params.temperature = -1.f;

  for (uint64_t seq = 0; seq < 8; ++seq) {
    EXPECT_EQ(sampleOnceDirect(logits, params, /*seed=*/42, /*seq=*/seq), 1);
  }
}

// -----------------------------------------------------------------------------
// 17. kernel::isGreedy() unit test — exercises all parameter combinations
// -----------------------------------------------------------------------------
TEST(sampler, isGreedy_function) {
  // Default: temperature=1, no knobs -> greedy.
  {
    kernel::SamplingParams p;
    EXPECT_TRUE(kernel::isGreedy(p));
  }
  // temperature <= 0 -> always greedy, regardless of other knobs.
  {
    kernel::SamplingParams p;
    p.temperature = 0.f;
    p.topK = 50;
    p.topP = 0.9f;
    p.minP = 0.1f;
    EXPECT_TRUE(kernel::isGreedy(p));
  }
  {
    kernel::SamplingParams p;
    p.temperature = -5.f;
    EXPECT_TRUE(kernel::isGreedy(p));
  }
  // temperature != 1 and > 0 -> not greedy.
  {
    kernel::SamplingParams p;
    p.temperature = 0.5f;
    EXPECT_FALSE(kernel::isGreedy(p));
  }
  {
    kernel::SamplingParams p;
    p.temperature = 2.f;
    EXPECT_FALSE(kernel::isGreedy(p));
  }
  // temperature == 1 but topK enabled -> not greedy.
  {
    kernel::SamplingParams p;
    p.temperature = 1.f;
    p.topK = 10;
    EXPECT_FALSE(kernel::isGreedy(p));
  }
  // temperature == 1 but topP enabled -> not greedy.
  {
    kernel::SamplingParams p;
    p.temperature = 1.f;
    p.topP = 0.9f;
    EXPECT_FALSE(kernel::isGreedy(p));
  }
  // temperature == 1 but minP enabled -> not greedy.
  {
    kernel::SamplingParams p;
    p.temperature = 1.f;
    p.minP = 0.05f;
    EXPECT_FALSE(kernel::isGreedy(p));
  }
  // Boundary: topP=1.0 (disabled), topP=0.0 (disabled by >0 check).
  {
    kernel::SamplingParams p;
    p.temperature = 1.f;
    p.topP = 1.0f;
    EXPECT_TRUE(kernel::isGreedy(p));
  }
  {
    kernel::SamplingParams p;
    p.temperature = 1.f;
    p.topP = 0.0f;
    EXPECT_TRUE(kernel::isGreedy(p));
  }
}

// -----------------------------------------------------------------------------
// 18. toKernelParams() conversion test
// -----------------------------------------------------------------------------
TEST(sampler, toKernelParams_conversion) {
  // Normal case: all fields set.
  {
    SamplerConfig cfg(0.7f, 50, 0.9f, 0.05f, 12345);
    auto p = toKernelParams(cfg);
    EXPECT_FLOAT_EQ(p.temperature, 0.7f);
    EXPECT_EQ(p.topK, 50);
    EXPECT_FLOAT_EQ(p.topP, 0.9f);
    EXPECT_FLOAT_EQ(p.minP, 0.05f);
    EXPECT_EQ(p.seed, 12345);
  }
  // Disabled knobs are normalized to their canonical disabled values.
  {
    SamplerConfig cfg(0.f, -1, 1.5f, -0.1f, -1);
    auto p = toKernelParams(cfg);
    EXPECT_FLOAT_EQ(p.temperature, 0.f);  // passed through
    EXPECT_EQ(p.topK, 0);                 // -1 -> 0
    EXPECT_FLOAT_EQ(p.topP, 1.f);         // 1.5 -> 1.0
    EXPECT_FLOAT_EQ(p.minP, 0.f);         // -0.1 -> 0.0
    EXPECT_EQ(p.seed, -1);                // passed through
  }
  // topP boundary: exactly 0 is "disabled" (the >0 check filters it).
  {
    SamplerConfig cfg(1.f, 0, 0.f, 0.f, -1);
    auto p = toKernelParams(cfg);
    EXPECT_FLOAT_EQ(p.topP, 1.f);  // 0.0 -> disabled
  }
}

}  // namespace tinygpt
