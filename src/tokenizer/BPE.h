/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <list>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <vector>

#include "Base.h"

namespace tinygpt::tokenizer {

class LRUCache {
 public:
  using Key = std::string;
  using Value = std::vector<int32_t>;

  explicit LRUCache(size_t capacity, size_t numSegments = 32);
  ~LRUCache() = default;

  std::optional<Value> get(const Key& key) const;
  void put(const Key& key, Value&& value) const;

  void erase(const Key& key) const;
  size_t size() const;

 private:
  struct Segment {
    using List = std::list<std::pair<Key, Value>>;
    using Map = ankerl::unordered_dense::map<Key, List::iterator>;

    explicit Segment(size_t cap) : capacity(cap) {}

    mutable std::shared_mutex mutex;
    List lru;
    Map map;
    size_t capacity;
  };

  size_t capacity_;
  size_t numSegments_;
  std::vector<std::unique_ptr<Segment>> segments_;

  Segment& segmentFor(const Key& key) const;
};

class BPE : public Component {
 public:
  BPE(const ankerl::unordered_dense::map<std::string, int32_t>& vocab,
      const ankerl::unordered_dense::map<StringPair, int32_t, StringPairHash>& merges, bool ignoreMerges = false,
      bool enableCache = false);

  ComponentType getType() override { return ComponentType::BPE; }

  int32_t token2Id(const std::string& token) override;
  std::string id2Token(int32_t id) override;
  std::vector<int32_t> tokenize(const PreTokenizedString& tokens) override;

 private:
  std::vector<std::string_view> bpeV1(std::string_view text);
  std::vector<std::string_view> bpeV2(std::string_view text);
  static std::vector<std::string_view> splitUTF8(std::string_view str);

  bool ignoreMerges_;
  bool enableCache_;
  ankerl::unordered_dense::map<std::string_view, int32_t> encoder_;
  ankerl::unordered_dense::map<int32_t, std::string> decoder_;
  ankerl::unordered_dense::map<StringViewPair, int32_t, StringViewPairHash> mergeRanks_;
  std::unique_ptr<LRUCache> cache_;

  std::string encoderBackStr_;
  std::string mergeRanksBackStr_;
};

}  // namespace tinygpt::tokenizer
