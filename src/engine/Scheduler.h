/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "Sampler.h"
#include "Tensor.h"
#include "engine/CUDAGraphRunner.h"
#include "model/GPTModel.h"
#include "tokenizer/Tokenizer.h"

namespace tinygpt {

class PagedKVCache;

enum class FinishReason {
  Stop,
  Length,
  Aborted,
};

struct GenRequest {
  std::string prompt;
  SamplerConfig samplerConfig;
  int32_t maxNewTokens = 0;
  std::vector<int32_t> stopTokenIds;

  std::function<bool(const std::string& tokenText)> onToken;
};

struct GenResult {
  std::vector<int32_t> promptIds;
  std::vector<int32_t> generatedIds;
  std::string text;
  FinishReason finishReason = FinishReason::Stop;
};

class Scheduler {
 public:
  Scheduler(GPTModel& model, PagedKVCache& cache, tokenizer::Tokenizer& tokenizer,
            const std::vector<int32_t>& baseEosTokenIds, int32_t maxBatchTokens = 8192, int32_t prefillChunkSize = 512,
            int32_t maxGraphBatch = 64);
  ~Scheduler();

  void start();
  void stop();

  std::shared_future<GenResult> submit(GenRequest req);

  // statistics
  size_t numWaiting() const;
  size_t numRunning() const { return runningCount_.load(); }

  // [0] command (0=stop, 1=run)
  // [1] totalTokens
  // [2] scheduledBatch
  // [3] maxSeqLenQ
  // [4] maxSeqLenKV
  // [5] i64Used
  // [6] i32Used
  // [7] reserved
  static constexpr int32_t kTPHeaderSize = 8;

 private:
  // per-request state
  struct GenSession {
    uint64_t kvSeqId = 0;
    std::vector<int32_t> promptIds;
    int32_t numCachedTokens = 0;
    int32_t numPromptForwarded = 0;
    std::vector<int32_t> generatedIds;
    int32_t lastToken = 0;
    Sampler sampler;
    std::vector<int32_t> stopTokenIds;
    int32_t maxNewTokens = 0;
    std::function<bool(const std::string&)> onToken;
    tokenizer::Tokenizer::StreamState streamState;
    bool finished = false;
    bool aborted = false;
    FinishReason finishReason = FinishReason::Length;
    std::shared_ptr<std::promise<GenResult>> promise;

    explicit GenSession(SamplerConfig cfg) : sampler(cfg) {}
  };

  void workerLoop();
  void runStep(std::vector<std::shared_ptr<GenSession>>& active);
  void harvestTokenIds();
  void processCallbacks();
  void completeSession(std::shared_ptr<GenSession>& session);
  void retireFinished(std::vector<std::shared_ptr<GenSession>>& active);
  void admitWaiting(std::vector<std::shared_ptr<GenSession>>& active);
  bool isStopToken(const GenSession& session, int32_t tokenId) const;

  void allocateMetaBuffers();

  void broadcastStepToWorkers(int64_t command, int64_t totalTokens, int64_t scheduledBatch, int64_t maxSeqLenQ,
                              int64_t maxSeqLenKV, int64_t i64Used, int64_t i32Used);

  void padMetadataForGraph(int32_t realBatch, int32_t graphBatch, int32_t realTokens);
  void captureAllGraphs();

  GPTModel& model_;
  PagedKVCache& cache_;
  tokenizer::Tokenizer& tokenizer_;
  std::vector<int32_t> baseEosTokenIds_;
  int32_t maxBatchTokens_;
  int32_t prefillChunkSize_;
  tinytorch::Device device_;

  tinytorch::Tensor splitKvO_;
  tinytorch::Tensor splitKvLse_;

  tinytorch::Tensor metaHostI64_;
  tinytorch::Tensor metaDevI64_;
  tinytorch::Tensor metaHostI32_;
  tinytorch::Tensor metaDevI32_;

  tinytorch::Tensor tpHeaderDev_;

  struct PrevStep {
    std::vector<std::shared_ptr<GenSession>> sessions;
    int32_t scheduledBatch = 0;
    bool hasPendingHarvest = false;
  };
  PrevStep prev_;

  struct PendingCallback {
    std::shared_ptr<GenSession> session;
    int32_t tokenId;
  };
  std::vector<PendingCallback> pendingCallbacks_;

  std::unique_ptr<BatchSampler> batchSampler_;

  std::vector<int32_t> graphBatchSizes_;
  std::unordered_map<int32_t, std::unique_ptr<CUDAGraphRunner>> graphRunners_;
  int32_t maxGraphBatch_ = 64;

  int32_t padKvBlockId_ = -1;

  mutable std::mutex waitMutex_;
  std::condition_variable waitCV_;
  std::deque<std::shared_ptr<GenSession>> waiting_;
  std::atomic<bool> running_{false};
  std::atomic<size_t> runningCount_{0};
  std::thread worker_;
};

}  // namespace tinygpt
