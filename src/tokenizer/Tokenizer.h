/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "BPE.h"
#include "Base.h"
#include "ByteLevel.h"
#include "Regex.h"
#include "Split.h"
#include "TemplateProcessing.h"

namespace tinygpt::tokenizer {

class Tokenizer {
 public:
  Tokenizer() = default;
  ~Tokenizer();

  bool initWithConfig(const std::string& tokenizerPath, const std::string& cfgPath);

  int32_t token2Id(const std::string& token);
  std::string id2Token(int32_t id);

  std::vector<int32_t> encode(const std::string& text, bool allowAddedTokens = true);
  std::vector<std::vector<int32_t>> encodeBatch(const std::vector<std::string>& texts, uint32_t numThreads = 8,
                                                bool allowAddedTokens = true);

  std::string decode(const std::vector<int32_t>& ids, int64_t offset = 0);
  std::vector<std::string> decodeBatch(const std::vector<std::vector<int32_t>>& ids, uint32_t numThreads = 8);

  // check whether the utf-8 sequence is complete, if not, return it on the next call.
  std::string decodeStream(const std::vector<int32_t>& ids);

  int32_t bosTokenId() const { return bosTokenId_; }
  int32_t eosTokenId() const { return eosTokenId_; }
  int32_t padTokenId() const { return padTokenId_; }

  std::string bosTokenStr() { return bosTokenId_ < 0 ? std::string() : id2Token(bosTokenId_); }
  std::string eosTokenStr() { return eosTokenId_ < 0 ? std::string() : id2Token(eosTokenId_); }
  std::string padTokenStr() { return padTokenId_ < 0 ? std::string() : id2Token(padTokenId_); }

 private:
  void addTokens(const ankerl::unordered_dense::map<std::string, int32_t>& tokens);
  std::vector<std::string> splitAddedTokens(const std::string& text) const;
  std::vector<int32_t> encodeWithModel(const std::string& text, bool addSpecialTokens) const;

  void workerThread();
  template <typename Input, typename Output, typename Func>
  void parallelFor(const std::vector<Input>& inputs, std::vector<Output>& outputs, Func func, uint32_t numThreads);

  std::unique_ptr<Component> normalizer_;
  std::unique_ptr<Component> preTokenizer_;
  std::unique_ptr<Component> model_;
  std::unique_ptr<Component> postProcessor_;
  std::unique_ptr<Component> decoder_;

  int32_t bosTokenId_ = -1;
  int32_t eosTokenId_ = -1;
  int32_t padTokenId_ = -1;

  bool addBosToken_ = false;
  bool addEosToken_ = false;

  // stream decode cache
  struct TokenStreamCache {
    int32_t id;
    size_t len;
  };
  std::vector<TokenStreamCache> streamCacheToken_;
  std::string streamCacheStr_;

  // added tokens
  std::string addedPattern_;
  std::unique_ptr<Regex> addedMatcher_;
  ankerl::unordered_dense::map<std::string, int32_t> addedEncoder_;
  int32_t minAddedTokenId_ = -1;
  int32_t maxAddedTokenId_ = -1;
  std::vector<std::string> addedDecoder_;

  // threads management
  std::vector<std::thread> threads_;
  moodycamel::ConcurrentQueue<std::function<void()>> tasks_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::atomic<bool> stop_{false};
};

}  // namespace tinygpt::tokenizer
