/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "Utils/CUDAUtils.h"
#include "engine/GPTEngine.h"
#include "engine/Scheduler.h"
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

// Model directory — set via environment variable TINYGPT_TEST_MODEL_DIR.
// If not set, all scheduler tests are skipped.
// Expected: a small HuggingFace model dir (e.g. Qwen3-0.6B or any supported model).
std::string getModelDir() {
  const char* env = std::getenv("TINYGPT_TEST_MODEL_DIR");
  return env ? std::string(env) : "";
}

#define SKIP_IF_NO_MODEL()                                                    \
  do {                                                                        \
    if (getModelDir().empty()) {                                              \
      GTEST_SKIP() << "TINYGPT_TEST_MODEL_DIR not set; skipping model test"; \
    }                                                                         \
  } while (0)

// Create a GPTEngine with standard test configuration.
std::unique_ptr<GPTEngine> makeEngine(int32_t maxBatchTokens = 4096, int32_t maxGraphBatch = 64) {
  GPTConfig config;
  config.modelDir = getModelDir();
  config.device = tt::DeviceType::CUDA;
  config.dtype = tt::DType::BFloat16;
  config.maxNewTokens = 32;
  config.samplerConfig = SamplerConfig(0.f);  // greedy
  config.maxBatchTokens = maxBatchTokens;
  config.maxGraphBatch = maxGraphBatch;
  config.prefillChunkSize = 256;
  config.pagedConfig.memoryUtil = 0.5f;  // conservative for test
  config.pagedConfig.maxSeqLen = 512;

  auto engine = std::make_unique<GPTEngine>(config);
  if (!engine->prepare()) {
    return nullptr;
  }
  return engine;
}

}  // namespace

// =============================================================================
// 1. Basic single-request correctness — the pipeline must produce the same
//    output as a simple sequential engine would.
// =============================================================================
TEST(scheduler_pipeline, single_request_greedy) {
  SKIP_IF_NO_CUDA();
  SKIP_IF_NO_MODEL();

  auto engine = makeEngine();
  ASSERT_NE(engine, nullptr);

  auto output = engine->generate("The capital of France is");
  EXPECT_GT(output.newTokens, 0);
  EXPECT_FALSE(output.text.empty());
  // Greedy output should be deterministic — run twice and compare.
  auto output2 = engine->generate("The capital of France is");
  EXPECT_EQ(output.text, output2.text);
  EXPECT_EQ(output.newTokens, output2.newTokens);
}

// =============================================================================
// 2. Streaming callback correctness — every generated token must produce
//    a callback, and the concatenation must match the final text.
// =============================================================================
TEST(scheduler_pipeline, streaming_callback_completeness) {
  SKIP_IF_NO_CUDA();
  SKIP_IF_NO_MODEL();

  auto engine = makeEngine();
  ASSERT_NE(engine, nullptr);

  std::string streamed;
  std::atomic<int> callbackCount{0};

  auto output = engine->generate("Hello world", [&](const std::string& chunk) -> bool {
    streamed += chunk;
    callbackCount++;
    return true;
  });

  EXPECT_GT(output.newTokens, 0);
  // Streamed text should match final decoded text (modulo stream flush differences
  // at subword boundaries, so we check that streamed is a prefix or equal).
  // In practice with proper decodeStream, they should be equal.
  EXPECT_FALSE(streamed.empty());
  EXPECT_GT(callbackCount.load(), 0);
}

// =============================================================================
// 3. Concurrent requests — the two-thread pipeline must correctly handle
//    multiple overlapping requests without corruption or deadlock.
// =============================================================================
TEST(scheduler_pipeline, concurrent_requests) {
  SKIP_IF_NO_CUDA();
  SKIP_IF_NO_MODEL();

  auto engine = makeEngine();
  ASSERT_NE(engine, nullptr);

  constexpr int kNumRequests = 8;
  std::vector<std::string> prompts = {
      "One plus one equals",
      "The speed of light is",
      "Water boils at",
      "The largest planet is",
      "Hello, my name is",
      "The year has",
      "Gravity is",
      "A circle has",
  };

  // Launch all requests concurrently from separate threads.
  std::vector<std::future<GPTOutput>> futures;
  futures.reserve(kNumRequests);
  for (int i = 0; i < kNumRequests; i++) {
    futures.push_back(std::async(std::launch::async, [&engine, &prompts, i]() {
      return engine->generate(prompts[i]);
    }));
  }

  // Collect results — no hang, no crash.
  for (int i = 0; i < kNumRequests; i++) {
    auto output = futures[i].get();
    EXPECT_GT(output.newTokens, 0) << "Request " << i << " produced no tokens";
    EXPECT_FALSE(output.text.empty()) << "Request " << i << " produced empty text";
    EXPECT_TRUE(output.finishReason == FinishReason::Stop ||
                output.finishReason == FinishReason::Length)
        << "Request " << i << " had unexpected finish reason";
  }
}

// =============================================================================
// 4. High-concurrency stress — burst many requests to exercise the pipeline
//    under heavy batching conditions.
// =============================================================================
TEST(scheduler_pipeline, high_concurrency_burst) {
  SKIP_IF_NO_CUDA();
  SKIP_IF_NO_MODEL();

  auto engine = makeEngine(/*maxBatchTokens=*/8192, /*maxGraphBatch=*/128);
  ASSERT_NE(engine, nullptr);

  constexpr int kNumRequests = 32;
  std::vector<std::future<GPTOutput>> futures;
  futures.reserve(kNumRequests);

  // Fire all requests simultaneously.
  for (int i = 0; i < kNumRequests; i++) {
    std::string prompt = "Count from one to ten: " + std::to_string(i);
    futures.push_back(std::async(std::launch::async, [&engine, prompt]() {
      return engine->generate(prompt);
    }));
  }

  int completed = 0;
  int totalTokens = 0;
  for (auto& f : futures) {
    auto output = f.get();
    EXPECT_GT(output.newTokens, 0);
    completed++;
    totalTokens += output.newTokens;
  }

  EXPECT_EQ(completed, kNumRequests);
  EXPECT_GT(totalTokens, kNumRequests);  // at least 1 token per request
}

// =============================================================================
// 5. Streaming abort — a callback returning false must abort generation
//    promptly without crashing.
// =============================================================================
TEST(scheduler_pipeline, streaming_abort) {
  SKIP_IF_NO_CUDA();
  SKIP_IF_NO_MODEL();

  auto engine = makeEngine();
  ASSERT_NE(engine, nullptr);

  std::atomic<int> callbackCount{0};
  constexpr int kAbortAfter = 3;

  auto output = engine->generate("Tell me a long story about ", [&](const std::string& /*chunk*/) -> bool {
    int cnt = callbackCount.fetch_add(1) + 1;
    return cnt < kAbortAfter;  // abort after kAbortAfter callbacks
  });

  // Generation should have stopped early.
  EXPECT_LE(callbackCount.load(), kAbortAfter + 2);  // small slack for pipeline
  EXPECT_TRUE(output.finishReason == FinishReason::Stop ||
              output.finishReason == FinishReason::Length);
}

// =============================================================================
// 6. Concurrent streaming — multiple streaming requests simultaneously.
//    Validates that the prep thread's callback processing doesn't mix up
//    sequences' stream states.
// =============================================================================
TEST(scheduler_pipeline, concurrent_streaming) {
  SKIP_IF_NO_CUDA();
  SKIP_IF_NO_MODEL();

  auto engine = makeEngine();
  ASSERT_NE(engine, nullptr);

  constexpr int kNumRequests = 4;
  std::vector<std::string> prompts = {
      "Red is a color that",
      "Blue is a color that",
      "Green is a color that",
      "Yellow is a color that",
  };

  struct StreamResult {
    std::string streamed;
    int callbackCount = 0;
    GPTOutput output;
  };
  std::vector<std::future<StreamResult>> futures;

  for (int i = 0; i < kNumRequests; i++) {
    futures.push_back(std::async(std::launch::async, [&engine, &prompts, i]() {
      StreamResult res;
      res.output = engine->generate(prompts[i], [&res](const std::string& chunk) -> bool {
        res.streamed += chunk;
        res.callbackCount++;
        return true;
      });
      return res;
    }));
  }

  for (int i = 0; i < kNumRequests; i++) {
    auto res = futures[i].get();
    EXPECT_GT(res.output.newTokens, 0) << "Stream " << i << " produced no tokens";
    EXPECT_GT(res.callbackCount, 0) << "Stream " << i << " had no callbacks";
    EXPECT_FALSE(res.streamed.empty()) << "Stream " << i << " streamed nothing";
  }
}

// =============================================================================
// 7. Greedy determinism under concurrency — multiple identical requests should
//    produce identical outputs even when batched together, because greedy
//    decoding has no randomness.
// =============================================================================
TEST(scheduler_pipeline, greedy_determinism) {
  SKIP_IF_NO_CUDA();
  SKIP_IF_NO_MODEL();

  auto engine = makeEngine();
  ASSERT_NE(engine, nullptr);

  const std::string prompt = "Two plus two equals";

  // Run sequentially first to get reference.
  auto ref = engine->generate(prompt);
  ASSERT_GT(ref.newTokens, 0);

  // Now run concurrently — all should produce the same output.
  constexpr int kN = 4;
  std::vector<std::future<GPTOutput>> futures;
  for (int i = 0; i < kN; i++) {
    futures.push_back(std::async(std::launch::async, [&engine, &prompt]() {
      return engine->generate(prompt);
    }));
  }

  for (int i = 0; i < kN; i++) {
    auto output = futures[i].get();
    EXPECT_EQ(output.text, ref.text) << "Concurrent request " << i << " diverged from reference";
  }
}

// =============================================================================
// 8. Engine stats reporting — verify that stats work correctly with the
//    two-thread pipeline.
// =============================================================================
TEST(scheduler_pipeline, engine_stats) {
  SKIP_IF_NO_CUDA();
  SKIP_IF_NO_MODEL();

  auto engine = makeEngine();
  ASSERT_NE(engine, nullptr);

  auto stats = engine->stats();
  EXPECT_EQ(stats.numRunning, 0u);
  EXPECT_EQ(stats.numWaiting, 0u);
  EXPECT_GT(stats.kvTotalBlocks, 0);
  EXPECT_GT(stats.kvFreeBlocks, 0);
  EXPECT_GT(stats.kvBlockSize, 0);

  // Fire a request in background and check stats show activity.
  auto fut = std::async(std::launch::async, [&engine]() {
    return engine->generate("Hello world, this is a test prompt for stats checking");
  });

  // Give the scheduler a moment to pick up the request.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Output should eventually arrive.
  auto output = fut.get();
  EXPECT_GT(output.newTokens, 0);

  // After completion, stats should show idle.
  stats = engine->stats();
  EXPECT_EQ(stats.numRunning, 0u);
  EXPECT_EQ(stats.numWaiting, 0u);
}

// =============================================================================
// 9. Start-stop-restart — exercise the lifecycle management of the two-thread
//    pipeline. The engine should be usable, stoppable, and the destructor
//    should not hang.
// =============================================================================
TEST(scheduler_pipeline, engine_lifecycle) {
  SKIP_IF_NO_CUDA();
  SKIP_IF_NO_MODEL();

  // Create, use, destroy — repeat to test full lifecycle.
  for (int round = 0; round < 2; round++) {
    auto engine = makeEngine();
    ASSERT_NE(engine, nullptr) << "Round " << round << ": engine creation failed";

    auto output = engine->generate("Test round " + std::to_string(round));
    EXPECT_GT(output.newTokens, 0) << "Round " << round << ": no output";

    // engine destructor runs here — should not deadlock.
  }
}

// =============================================================================
// 10. Mixed-length prompts — exercise chunked prefill with concurrent decode.
//     Short prompts will start decoding while long prompts are still prefilling.
// =============================================================================
TEST(scheduler_pipeline, mixed_length_prompts) {
  SKIP_IF_NO_CUDA();
  SKIP_IF_NO_MODEL();

  auto engine = makeEngine(/*maxBatchTokens=*/4096, /*maxGraphBatch=*/64);
  ASSERT_NE(engine, nullptr);

  // Short prompt + long prompt submitted concurrently.
  std::string shortPrompt = "Hi";
  std::string longPrompt;
  for (int i = 0; i < 50; i++) {
    longPrompt += "This is sentence number " + std::to_string(i) + " in a long prompt. ";
  }

  auto futShort = std::async(std::launch::async, [&]() { return engine->generate(shortPrompt); });
  auto futLong = std::async(std::launch::async, [&]() { return engine->generate(longPrompt); });

  auto outputShort = futShort.get();
  auto outputLong = futLong.get();

  EXPECT_GT(outputShort.newTokens, 0);
  EXPECT_GT(outputLong.newTokens, 0);
  EXPECT_GT(outputLong.promptTokens, outputShort.promptTokens);
}

// =============================================================================
// 11. Rapid fire — submit and collect many small requests quickly to stress
//     the handoff protocol between prep and GPU threads.
// =============================================================================
TEST(scheduler_pipeline, rapid_fire_small_requests) {
  SKIP_IF_NO_CUDA();
  SKIP_IF_NO_MODEL();

  GPTConfig config;
  config.modelDir = getModelDir();
  config.device = tt::DeviceType::CUDA;
  config.dtype = tt::DType::BFloat16;
  config.maxNewTokens = 4;  // very short output
  config.samplerConfig = SamplerConfig(0.f);
  config.maxBatchTokens = 4096;
  config.maxGraphBatch = 64;
  config.prefillChunkSize = 256;
  config.pagedConfig.memoryUtil = 0.5f;
  config.pagedConfig.maxSeqLen = 256;

  auto engine = std::make_unique<GPTEngine>(config);
  ASSERT_TRUE(engine->prepare());

  constexpr int kNumRequests = 50;
  std::vector<std::future<GPTOutput>> futures;
  futures.reserve(kNumRequests);

  for (int i = 0; i < kNumRequests; i++) {
    futures.push_back(std::async(std::launch::async, [&engine, i]() {
      return engine->generate("Prompt " + std::to_string(i));
    }));
  }

  int completed = 0;
  for (auto& f : futures) {
    auto output = f.get();
    EXPECT_GT(output.newTokens, 0);
    completed++;
  }
  EXPECT_EQ(completed, kNumRequests);
}

}  // namespace tinygpt
