/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "Tokenizer.h"

#include <cassert>
#include <thread>

namespace tinygpt::tokenizer {

Tokenizer::~Tokenizer() {
  stop_ = true;
  cv_.notify_all();
  for (auto& t : threads_) {
    if (t.joinable()) {
      t.join();
    }
  }
}

bool Tokenizer::initWithConfigHF(const std::string& tokenizerPath, const std::string& cfgPath) {
  huggingface::ConfigHuggingface config;
  if (!huggingface::load(config, tokenizerPath, cfgPath)) {
    LOGE("load huggingface config failed.");
    return false;
  }

  normalizer_ = huggingface::createComponent(config.normalizer);
  preTokenizer_ = huggingface::createComponent(config.preTokenizer);
  model_ = huggingface::createComponent(config.model);
  postProcessor_ = huggingface::createComponent(config.postProcessor);
  decoder_ = huggingface::createComponent(config.decoder);

  // add token
  ankerl::unordered_dense::map<std::string, int32_t> addedTokens;
  for (auto& t : config.addedTokens) {
    // skip reserved tokens
    constexpr char const* RESERVED_TOKEN_HF = "reserved_special_token";
    if (t.content.find(RESERVED_TOKEN_HF) != std::string::npos) {
      continue;
    }
    addedTokens[t.content] = t.id;
  }
  addTokens(addedTokens);

  addBosToken_ = config.addBosToken;
  addEosToken_ = config.addEosToken;

  bosTokenId_ = token2Id(config.bosToken.content);
  eosTokenId_ = token2Id(config.eosToken.content);
  padTokenId_ = token2Id(config.padToken.content);

  assert(addBosToken_ || bosTokenId_ >= 0);
  assert(addEosToken_ || eosTokenId_ >= 0);
  return true;
}

bool Tokenizer::initWithConfigGPT2(const std::string& encoderPath, const std::string& vocabPath) {
  gpt2::ConfigGPT2 config;
  if (!gpt2::load(config, encoderPath, vocabPath)) {
    LOGE("load gpt2 config failed.");
    return false;
  }

  preTokenizer_ = std::make_unique<ByteLevel>(false, true);
  model_ = std::make_unique<BPE>(config.vocab, config.merges, false);
  decoder_ = std::make_unique<ByteLevel>();

  addBosToken_ = false;
  addEosToken_ = false;
  return true;
}

int32_t Tokenizer::token2Id(const std::string& token) {
  auto it = addedEncoder_.find(token);
  if (it != addedEncoder_.end()) {
    return it->second;
  }
  return model_->token2Id(token);
}

std::string Tokenizer::id2Token(int32_t id) {
  auto it = addedDecoder_.find(id);
  if (it != addedDecoder_.end()) {
    return it->second;
  }
  return model_->id2Token(id);
}

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool allowAddedTokens) {
  std::vector<int32_t> ret;
  if (addBosToken_) {
    ret.push_back(bosTokenId_);
  }

  if (!allowAddedTokens) {
    auto ids = encodeOrdinary(text);
    ret.insert(ret.end(), ids.begin(), ids.end());
  } else {
    auto pieces = splitAddedTokens(text);
    for (auto& piece : pieces) {
      // added tokens
      auto it = addedEncoder_.find(piece);
      if (it != addedEncoder_.end()) {
        ret.push_back(it->second);
        continue;
      }

      // other tokens
      auto ids = encodeOrdinary(piece);
      ret.insert(ret.end(), ids.begin(), ids.end());
    }
  }

  if (addEosToken_) {
    ret.push_back(eosTokenId_);
  }
  return ret;
}

std::vector<std::vector<int32_t>> Tokenizer::encodeBatch(const std::vector<std::string>& texts, uint32_t numThreads,
                                                         bool allowAddedTokens) {
  std::vector<std::vector<int32_t>> results(texts.size());
  parallelFor<std::string, std::vector<int32_t>>(
      texts, results, [&](const std::string& s) { return encode(s, allowAddedTokens); }, numThreads);
  return results;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) {
  std::vector<std::string> pieces;
  pieces.reserve(ids.size());

  for (auto& id : ids) {
    pieces.push_back(id2Token(id));
  }
  return decoder_->decode(pieces);
}

std::vector<std::string> Tokenizer::decodeBatch(const std::vector<std::vector<int32_t>>& ids, uint32_t numThreads) {
  std::vector<std::string> results(ids.size());
  parallelFor<std::vector<int32_t>, std::string>(
      ids, results, [&](const std::vector<int32_t>& v) { return decode(v); }, numThreads);
  return results;
}

void Tokenizer::addTokens(const ankerl::unordered_dense::map<std::string, int32_t>& tokens) {
  addedEncoder_ = tokens;
  int32_t idx = 0;
  for (auto& [k, v] : tokens) {
    addedDecoder_[v] = k;
    if (idx > 0) addedPattern_ += "|";
    addedPattern_ += re2::RE2::QuoteMeta(k);
    idx++;
  }
  if (!addedPattern_.empty()) {
    addedMatchers_.reserve(NUM_MAX_THREAD);
    for (uint32_t i = 0; i < NUM_MAX_THREAD; i++) {
      addedMatchers_.emplace_back(std::make_unique<re2::RE2>("(" + addedPattern_ + ")"));
    }
    assert(addedMatchers_[0]->ok());
  }
}

std::vector<std::string> Tokenizer::splitAddedTokens(const std::string& text) const {
  if (addedMatchers_.empty()) {
    return {text};
  }
  const auto tId = std::hash<std::thread::id>{}(std::this_thread::get_id());
  auto& matcher = *addedMatchers_[tId % NUM_MAX_THREAD];

  std::vector<std::string> result;
  re2::StringPiece input(text);
  size_t lastPos = 0;
  re2::StringPiece match;
  while (RE2::FindAndConsume(&input, matcher, &match)) {
    size_t matchPos = match.data() - text.data();
    if (lastPos < matchPos) {
      result.emplace_back(text.substr(lastPos, matchPos - lastPos));
    }
    result.emplace_back(match.data(), match.size());
    lastPos = matchPos + match.size();
  }
  if (lastPos < text.size()) {
    result.emplace_back(text.substr(lastPos));
  }
  return result;
}

std::vector<int32_t> Tokenizer::encodeOrdinary(const std::string& text) const {
  std::string normedText;
  if (normalizer_) {
    normedText = normalizer_->normalize(text);
  } else {
    normedText = text;
  }
  PreTokenizedString preTokenizedStr;
  if (preTokenizer_) {
    preTokenizedStr = preTokenizer_->preTokenize(normedText);
  } else {
    preTokenizedStr.backStr = normedText;
    preTokenizedStr.pieces = {{0, normedText.size() - 1}};
  }
  std::vector<int32_t> ids = model_->tokenize(preTokenizedStr);
  if (postProcessor_) {
    ids = postProcessor_->postProcess(ids);
  }
  return ids;
}

void Tokenizer::workerThread() {
  while (!stop_) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
      if (stop_ && tasks_.empty()) return;
      task = std::move(tasks_.front());
      tasks_.pop();
    }
    task();
  }
}

template <typename Input, typename Output, typename Func>
void Tokenizer::parallelFor(const std::vector<Input>& inputs, std::vector<Output>& outputs, Func func,
                            uint32_t numThreads) {
  size_t n = inputs.size();
  if (n == 0) {
    return;
  }
  if (numThreads >= NUM_MAX_THREAD) {
    LOGE("invalid numThreads, maximum: %d", NUM_MAX_THREAD);
    numThreads = NUM_MAX_THREAD;
  }
  numThreads = std::min<uint32_t>(numThreads, n);

  {
    // init threads
    std::lock_guard<std::mutex> lock(mutex_);
    if (threads_.size() < numThreads) {
      for (uint32_t i = threads_.size(); i < numThreads; i++) {
        threads_.emplace_back(&Tokenizer::workerThread, this);
      }
    }
  }

  std::atomic<size_t> finished{0};
  size_t batch = n / numThreads, rem = n % numThreads;
  size_t start = 0;

  for (uint32_t t = 0; t < numThreads; t++) {
    size_t end = start + batch + (t < rem ? 1 : 0);
    auto task = [&, start, end]() {
      for (size_t i = start; i < end; i++) {
        outputs[i] = func(inputs[i]);
      }
      ++finished;
    };
    {
      std::lock_guard<std::mutex> lock(mutex_);
      tasks_.emplace(task);
    }
    cv_.notify_one();
    start = end;
  }

  // wait until all tasks done
  while (finished.load() < numThreads) {
    std::this_thread::yield();
  }
}

}  // namespace tinygpt::tokenizer