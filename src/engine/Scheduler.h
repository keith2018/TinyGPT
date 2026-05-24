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

 private:
  struct Seq {
    uint64_t cacheId = 0;
    std::vector<int32_t> promptIds;
    int32_t pastLen = 0;
    int32_t numPromptProcessed = 0;
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

    explicit Seq(SamplerConfig cfg) : sampler(cfg) {}
  };

  void workerLoop();
  void runStep(std::vector<std::shared_ptr<Seq>>& active);
  void harvestTokenIds();
  void processCallbacks();
  void completeSeq(std::shared_ptr<Seq>& seq);
  void sweepFinished(std::vector<std::shared_ptr<Seq>>& active);
  bool admitWaiting(std::vector<std::shared_ptr<Seq>>& active);
  bool isEos(const Seq& seq, int32_t tokenId) const;

  void allocateMetaBuffers();

  void padMetadataForGraph(int32_t realBatch, int32_t graphBatch, int32_t realTokens);
  void captureAllGraphs();

  GPTModel& model_;
  PagedKVCache& cache_;
  tokenizer::Tokenizer& tokenizer_;
  std::vector<int32_t> baseEosTokenIds_;
  int32_t maxBatchTokens_;
  int32_t prefillChunkSize_;
  tinytorch::Device device_;

  // pre-allocated split-kv tmp buffers
  tinytorch::Tensor tmpO_;
  tinytorch::Tensor tmpLse_;

  tinytorch::Tensor hostI64_;      // pinned CPU, Int64
  tinytorch::Tensor devI64_;       // CUDA, Int64
  tinytorch::Tensor hostI32_;      // pinned CPU, Int32
  tinytorch::Tensor devI32_;       // CUDA, Int32
  tinytorch::Tensor sampledHost_;  // Int64 [maxBatchTokens_]

  struct PrevStep {
    std::vector<std::shared_ptr<Seq>> seqs;
    int32_t scheduledBatch = 0;
    bool valid = false;
  };
  PrevStep prev_;

  // deferred streaming callbacks — filled by harvestTokenIds(), drained by processCallbacks()
  struct PendingCallback {
    std::shared_ptr<Seq> seq;
    int32_t tokenId;
  };
  std::vector<PendingCallback> pendingCallbacks_;

  struct CudaPipeState;
  std::unique_ptr<CudaPipeState> cudaPipe_;

  // cuda graph runners — captures at batch sizes [1,2,4,8,16,24,...,maxGraphBatch_]
  std::vector<int32_t> graphBatchSizes_;
  std::unordered_map<int32_t, std::unique_ptr<CUDAGraphRunner>> graphRunners_;
  int32_t maxGraphBatch_ = 64;

  // dummy KV block: padding sequences scatter KV here to avoid corrupting real data
  int32_t dummyBlockId_ = -1;

  mutable std::mutex waitMutex_;
  std::condition_variable waitCV_;
  std::deque<std::shared_ptr<Seq>> waiting_;
  std::atomic<bool> running_{false};
  std::atomic<size_t> runningCount_{0};
  std::thread worker_;
};

}  // namespace tinygpt
