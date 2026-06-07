/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "GPTEngine.h"

#include <utility>

#include "Utils/CUDAUtils.h"
#include "distributed/Communicator.h"

namespace tt = tinytorch;

namespace tinygpt {

GPTEngine::GPTEngine(GPTConfig config) : config_(std::move(config)) {}

GPTEngine::~GPTEngine() {
  if (scheduler_) {
    scheduler_->stop();
  }
}

bool GPTEngine::prepare() {
  ASSERT(config_.device.isCuda());

  huggingface::ModelLoader loader;
  if (!loader.load(config_.modelDir, config_.device, config_.dtype)) {
    LOGE("Prepare failed: model load");
    return false;
  }
  context_ = loader.getContext();

  if (context_.generationConfig) {
    for (auto id : context_.generationConfig->eosTokenIds) {
      baseEosTokenIds_.push_back(id);
    }
  }
  if (baseEosTokenIds_.empty()) {
    int32_t eosId = context_.tokenizer->eosTokenId();
    if (eosId >= 0) {
      baseEosTokenIds_.push_back(eosId);
    }
  }

  auto sizing = PagedKVCache::autoSize(*context_.model, config_.dtype, config_.pagedConfig);
  // under tp, each rank only holds numKvHeads/worldSize of kv cache
  const int worldSize = distributed::Communicator::tp().worldSize();
  const int64_t localKvHeads = context_.model->numKvHeads() / worldSize;
  ASSERT(context_.model->numKvHeads() % worldSize == 0);

  // broadcast rank0's numBlocks so all ranks agree
  if (distributed::Communicator::tp().enabled()) {
    auto opts = tt::Options(config_.device, tt::DType::Int64).noGrad();
    auto numBlocksDev = tt::Tensor({1}, opts);
    int64_t hostVal = sizing.numBlocks;
    auto& stream = tt::cuda::getCurrentCUDAStream(config_.device.index);
    tt::Storage::copyOnDevice(numBlocksDev.dataPtr<>(), config_.device, &hostVal, tt::Device::cpu(), sizeof(int64_t),
                              &stream);
    distributed::Communicator::tp().broadcast(numBlocksDev);
    tt::Storage::copyOnDevice(&hostVal, tt::Device::cpu(), numBlocksDev.dataPtr<>(), config_.device, sizeof(int64_t),
                              &stream);
    stream.synchronize();
    sizing.numBlocks = hostVal;
  }

  pagedCache_ = std::make_unique<PagedKVCache>(context_.model->numLayers(), localKvHeads, context_.model->headDim(),
                                               sizing, tt::Options(config_.device, config_.dtype));
  context_.model->setPagedCache(pagedCache_.get());

  // only rank 0 owns the scheduler; other ranks run WorkerRuntime externally.
  if (distributed::Communicator::tp().rank() == 0) {
    scheduler_ = std::make_unique<Scheduler>(*context_.model, *pagedCache_, *context_.tokenizer, baseEosTokenIds_,
                                             config_.maxBatchTokens, config_.prefillChunkSize, config_.maxGraphBatch);
    scheduler_->start();
  }
  return true;
}

void GPTEngine::reconfigure(const SamplerConfig& samplerConfig, int32_t maxNewTokens,
                            const std::vector<int32_t>& extraStopTokenIds) {
  config_.samplerConfig = samplerConfig;
  config_.maxNewTokens = maxNewTokens;
  extraEosTokenIds_.clear();
  for (auto id : extraStopTokenIds) {
    // only add if not already present
    if (std::find(baseEosTokenIds_.begin(), baseEosTokenIds_.end(), id) == baseEosTokenIds_.end() &&
        std::find(extraEosTokenIds_.begin(), extraEosTokenIds_.end(), id) == extraEosTokenIds_.end()) {
      extraEosTokenIds_.push_back(id);
    }
  }
}

bool GPTEngine::hasChatTemplate() const { return context_.tokenizer && context_.tokenizer->hasChatTemplate(); }

std::string GPTEngine::applyChatTemplate(const std::vector<tokenizer::ChatMessage>& messages,
                                         bool addGenerationPrompt) const {
  if (!context_.tokenizer) return {};
  return context_.tokenizer->applyChatTemplate(messages, addGenerationPrompt);
}

GPTEngine::EngineStats GPTEngine::stats() const {
  EngineStats s;
  if (scheduler_) {
    s.numRunning = scheduler_->numRunning();
    s.numWaiting = scheduler_->numWaiting();
  }
  if (pagedCache_) {
    s.kvTotalBlocks = pagedCache_->numTotalBlocks();
    s.kvFreeBlocks = pagedCache_->numFreeBlocks();
    s.kvBlockSize = pagedCache_->blockSize();
  }
  return s;
}

GPTOutput GPTEngine::generate(const std::string& prompt, const GenerateCallback& callback) {
  return generate(prompt, config_.samplerConfig, config_.maxNewTokens, extraEosTokenIds_, callback);
}

GPTOutput GPTEngine::generate(const std::string& prompt, const SamplerConfig& samplerConfig, int32_t maxNewTokens,
                              const std::vector<int32_t>& extraStopTokenIds, const GenerateCallback& callback) {
  GenRequest req;
  req.prompt = prompt;
  req.samplerConfig = samplerConfig;
  req.maxNewTokens = maxNewTokens;
  req.stopTokenIds = extraStopTokenIds;
  req.onToken = callback;

  auto fut = scheduler_->submit(std::move(req));
  auto result = fut.get();

  GPTOutput output;
  output.tokenIds.reserve(result.promptIds.size() + result.generatedIds.size());
  output.tokenIds.insert(output.tokenIds.end(), result.promptIds.begin(), result.promptIds.end());
  output.tokenIds.insert(output.tokenIds.end(), result.generatedIds.begin(), result.generatedIds.end());
  output.text = std::move(result.text);
  output.promptTokens = static_cast<int32_t>(result.promptIds.size());
  output.newTokens = static_cast<int32_t>(result.generatedIds.size());
  output.finishReason = result.finishReason;
  return output;
}

}  // namespace tinygpt