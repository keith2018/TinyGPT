/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "Tokenizer.h"

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
  if (token.empty()) {
    return -1;
  }
  auto it = addedEncoder_.find(token);
  if (it != addedEncoder_.end()) {
    return it->second;
  }
  return model_->token2Id(token);
}

std::string Tokenizer::id2Token(int32_t id) {
  if (id >= minAddedTokenId_ && id <= maxAddedTokenId_) {
    return addedDecoder_[id - minAddedTokenId_];
  }
  return model_->id2Token(id);
}

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool allowAddedTokens) {
  std::vector<int32_t> ret;
  if (addBosToken_) {
    ret.push_back(bosTokenId_);
  }

  if (!allowAddedTokens) {
    auto ids = encodeWithModel(text, false);
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
      auto ids = encodeWithModel(piece, true);
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

std::string Tokenizer::decodeStream(const std::vector<int32_t>& ids) {
  size_t totalNewLen = 0;
  std::vector<std::string> newTokens;
  newTokens.reserve(ids.size());

  // decode new ids
  for (auto& id : ids) {
    std::string tokenStr = id2Token(id);
    totalNewLen += tokenStr.size();
    newTokens.push_back(std::move(tokenStr));
  }

  // add to cache
  streamCacheToken_.reserve(streamCacheToken_.size() + ids.size());
  streamCacheStr_.reserve(streamCacheStr_.size() + totalNewLen);
  for (size_t i = 0; i < ids.size(); i++) {
    streamCacheToken_.push_back({ids[i], newTokens[i].size()});
    streamCacheStr_ += newTokens[i];
  }

  // check utf8 complete
  auto incompletePos = ByteLevel::findIncompletePos(streamCacheStr_);
  if (incompletePos < 0) {
    streamCacheToken_.clear();
    std::string retStr = std::move(streamCacheStr_);
    streamCacheStr_.clear();
    return retStr;
  }

  size_t incompleteLen = streamCacheStr_.size() - incompletePos;
  size_t totalIncompleteLen = 0;
  auto keepTokens = static_cast<int32_t>(streamCacheToken_.size());
  while (keepTokens > 0) {
    totalIncompleteLen += streamCacheToken_[keepTokens - 1].len;
    if (totalIncompleteLen >= incompleteLen) {
      break;
    }
    keepTokens--;
  }

  if (keepTokens <= 0) {
    streamCacheToken_.clear();
    streamCacheStr_.clear();
    return {};
  }

  // adjust cache
  streamCacheToken_.resize(keepTokens);

  auto splitPos = streamCacheStr_.size() - totalIncompleteLen;
  auto retStr = streamCacheStr_.substr(0, splitPos);
  streamCacheStr_ = streamCacheStr_.substr(splitPos);
  return retStr;
}

void Tokenizer::addTokens(const ankerl::unordered_dense::map<std::string, int32_t>& tokens) {
  addedEncoder_ = tokens;
  minAddedTokenId_ = std::numeric_limits<int32_t>::max();
  int32_t cnt = 0;
  for (auto& [k, v] : tokens) {
    minAddedTokenId_ = std::min(minAddedTokenId_, v);
    maxAddedTokenId_ = std::max(maxAddedTokenId_, v);
    if (cnt > 0) {
      addedPattern_ += "|";
    }
    addedPattern_ += Regex::quoteMeta(k);
    cnt++;
  }
  addedDecoder_.resize(maxAddedTokenId_ - minAddedTokenId_ + 1);
  for (auto& [k, v] : tokens) {
    addedDecoder_[v - minAddedTokenId_] = k;
  }
  if (!addedPattern_.empty()) {
    addedMatcher_ = std::make_unique<Regex>("(" + addedPattern_ + ")");
    assert(addedMatcher_->valid());
  }
}

std::vector<std::string> Tokenizer::splitAddedTokens(const std::string& text) const {
  if (!addedMatcher_) {
    return {text};
  }
  const auto ranges = Split::split(text, *addedMatcher_, SplitDelimiterBehavior::ISOLATED);
  std::vector<std::string> results;
  results.reserve(ranges.size());
  for (const auto& r : ranges) {
    results.emplace_back(text.substr(r.first, r.second - r.first));
  }
  return results;
}

std::vector<int32_t> Tokenizer::encodeWithModel(const std::string& text, bool addSpecialTokens) const {
  std::string normedText;
  if (normalizer_) {
    normedText = normalizer_->normalize(text);
  } else {
    normedText = text;
  }
  StringPieces preTokenizedStr(normedText);
  if (preTokenizer_) {
    preTokenizedStr = preTokenizer_->preTokenize(preTokenizedStr);
  }
  std::vector<int32_t> ids = model_->tokenize(preTokenizedStr);
  if (postProcessor_) {
    ids = postProcessor_->postProcess(ids, addSpecialTokens);
  }
  return ids;
}

void Tokenizer::workerThread() {
  while (!stop_) {
    std::function<void()> task;
    if (tasks_.try_dequeue(task)) {
      task();
    } else {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] { return stop_ || tasks_.size_approx() > 0; });
    }
  }
}

template <typename Input, typename Output, typename Func>
void Tokenizer::parallelFor(const std::vector<Input>& inputs, std::vector<Output>& outputs, Func func,
                            uint32_t numThreads) {
  size_t n = inputs.size();
  if (n == 0) {
    return;
  }

  static auto maxThreads = std::thread::hardware_concurrency();
  numThreads = std::min<uint32_t>(numThreads, maxThreads);
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
  std::mutex finishedMutex;
  std::condition_variable finishedCv;

  std::vector<std::function<void()>> fs;
  fs.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    auto task = [&, i]() {
      outputs[i] = func(inputs[i]);
      if (++finished == inputs.size()) {
        std::lock_guard<std::mutex> lock(finishedMutex);
        finishedCv.notify_one();
      }
    };
    fs.emplace_back(task);
  }
  tasks_.enqueue_bulk(fs.begin(), fs.size());
  cv_.notify_all();

  // wait until all tasks done
  std::unique_lock<std::mutex> lock(finishedMutex);
  finishedCv.wait(lock, [&] { return finished == inputs.size(); });
}

}  // namespace tinygpt::tokenizer