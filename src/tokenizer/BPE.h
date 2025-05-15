/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <list>
#include <optional>
#include <string>
#include <vector>

#include "Base.h"

namespace tinygpt::tokenizer {

class LRUCache {
 public:
  using Key = std::string;
  using Value = std::vector<int32_t>;
  using ListIt = std::list<Key>::iterator;

  explicit LRUCache(size_t capacity = NUM_MAX_CACHE);
  ~LRUCache() = default;

  std::optional<Value> get(const Key& key) const;
  void put(const Key& key, Value&& value) const;

  void erase(const Key& key) const;
  size_t size() const;

 private:
  size_t capacity_;
  mutable std::list<Key> list_;
  mutable ankerl::unordered_dense::map<Key, std::pair<Value, ListIt>> map_;
};

class BPE : public Component {
 public:
  BPE(const ankerl::unordered_dense::map<std::string, int32_t>& vocab,
      const ankerl::unordered_dense::map<StringPair, int32_t, StringPairHash>& merges, bool ignoreMerges = false,
      bool enableCache = true);

  ComponentType getType() override { return ComponentType::BPE; }

  int32_t token2Id(const std::string& token) override;
  std::string id2Token(int32_t id) override;
  std::vector<int32_t> tokenize(const StringPieces& tokens) override;

 private:
  std::vector<std::string_view> bpeV1(std::string_view text);
  std::vector<std::string_view> bpeV2(std::string_view text);

  bool ignoreMerges_;
  bool enableCache_;
  ankerl::unordered_dense::map<std::string_view, int32_t> encoder_;
  ankerl::unordered_dense::map<int32_t, std::string> decoder_;
  ankerl::unordered_dense::map<StringViewPair, int32_t, StringViewPairHash> mergeRanks_;

  std::string encoderBackStr_;
  std::string mergeRanksBackStr_;
};

}  // namespace tinygpt::tokenizer
