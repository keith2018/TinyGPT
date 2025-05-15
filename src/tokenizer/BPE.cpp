/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "BPE.h"

#include "ByteLevel.h"

namespace tinygpt::tokenizer {

LRUCache::LRUCache(size_t capacity, size_t numSegments) : capacity_(capacity), numSegments_(numSegments) {
  segments_.resize(numSegments);
  size_t perSegment = (capacity + numSegments - 1) / numSegments;
  for (auto& seg : segments_) {
    seg = std::make_unique<Segment>(perSegment);
  }
}

std::optional<LRUCache::Value> LRUCache::get(const Key& key) const {
  auto& seg = segmentFor(key);
  std::unique_lock lock(seg.mutex);
  auto it = seg.map.find(key);
  if (it == seg.map.end()) return std::nullopt;
  seg.lru.splice(seg.lru.begin(), seg.lru, it->second);
  return it->second->second;
}

void LRUCache::put(const Key& key, Value&& value) const {
  auto& seg = segmentFor(key);
  std::unique_lock lock(seg.mutex);
  auto it = seg.map.find(key);
  if (it != seg.map.end()) {
    it->second->second = std::move(value);
    seg.lru.splice(seg.lru.begin(), seg.lru, it->second);
  } else {
    seg.lru.emplace_front(key, std::move(value));
    seg.map[seg.lru.front().first] = seg.lru.begin();
    if (seg.lru.size() > seg.capacity) {
      auto last = seg.lru.end();
      --last;
      seg.map.erase(last->first);
      seg.lru.pop_back();
    }
  }
}

void LRUCache::erase(const Key& key) const {
  auto& seg = segmentFor(key);
  std::unique_lock lock(seg.mutex);
  auto it = seg.map.find(key);
  if (it != seg.map.end()) {
    seg.lru.erase(it->second);
    seg.map.erase(it);
  }
}

size_t LRUCache::size() const {
  size_t total = 0;
  for (const auto& segPtr : segments_) {
    std::shared_lock lock(segPtr->mutex);
    total += segPtr->map.size();
  }
  return total;
}

LRUCache::Segment& LRUCache::segmentFor(const Key& key) const {
  size_t h = ankerl::unordered_dense::hash<Key>{}(key);
  return *segments_[h % numSegments_];
}

BPE::BPE(const ankerl::unordered_dense::map<std::string, int32_t>& vocab,
         const ankerl::unordered_dense::map<StringPair, int32_t, StringPairHash>& merges, bool ignoreMerges,
         bool enableCache)
    : ignoreMerges_(ignoreMerges), enableCache_(enableCache) {
  // encoder_ & decoder_
  size_t encoderStrLen = 0;
  for (auto& [k, v] : vocab) {
    encoderStrLen += k.size();
  }
  encoderBackStr_.reserve(encoderStrLen);
  encoder_.reserve(vocab.size());
  decoder_.reserve(vocab.size());
  for (auto& [k, v] : vocab) {
    const auto* ptr = encoderBackStr_.data() + encoderBackStr_.size();
    encoderBackStr_.append(k);
    encoder_[std::string_view(ptr, k.size())] = v;
    decoder_[v] = ByteLevel::utf8ToBytes(k);
  }

  // mergeRanks_
  size_t mergeStrLen = 0;
  for (auto& [k, v] : merges) {
    mergeStrLen += k.first.size() + k.second.size();
  }
  mergeRanksBackStr_.reserve(mergeStrLen);
  mergeRanks_.reserve(merges.size());
  for (auto& [k, v] : merges) {
    const auto* ptr1 = mergeRanksBackStr_.data() + mergeRanksBackStr_.size();
    const auto* ptr2 = ptr1 + k.first.size();
    mergeRanksBackStr_.append(k.first);
    mergeRanksBackStr_.append(k.second);
    mergeRanks_[{std::string_view(ptr1, k.first.size()), std::string_view(ptr2, k.second.size())}] = v;
  }

  // cache
  if (enableCache) {
    cache_ = std::make_unique<LRUCache>();
  }
}

int32_t BPE::token2Id(const std::string& token) {
  auto it = encoder_.find(token);
  if (it != encoder_.end()) {
    return it->second;
  }

  LOGE("error encode token: %s", token.c_str());
  return -1;
}

std::string BPE::id2Token(int32_t id) {
  auto it = decoder_.find(id);
  if (it != decoder_.end()) {
    return it->second;
  }

  LOGE("error decode id: %d", id);
  return {};
}

std::vector<int32_t> BPE::tokenize(const StringPieces& tokens) {
  std::vector<int32_t> ret;
  auto reserveSize = static_cast<float>(tokens.pieces.size()) * 1.5;
  ret.reserve(static_cast<size_t>(reserveSize));

  for (auto& piece : tokens.pieces) {
    auto token = tokens.backStr.substr(piece.first, piece.second - piece.first);
    // ignore merge
    if (ignoreMerges_) {
      auto it = encoder_.find(token);
      if (it != encoder_.end()) {
        ret.push_back(it->second);
        continue;
      }
    }

    // cache
    if (enableCache_) {
      auto cacheItem = cache_->get(token);
      if (cacheItem) {
        ret.insert(ret.end(), cacheItem.value().begin(), cacheItem.value().end());
        continue;
      }
    }

    // bpe
    std::vector<std::string_view> bpePieces = bpeV2(token);
    std::vector<int32_t> ids;
    ids.reserve(bpePieces.size());
    for (auto& bpePiece : bpePieces) {
      auto it = encoder_.find(bpePiece);
      if (it != encoder_.end()) {
        ids.push_back(it->second);
      } else {
        LOGE("error encode token: %s", std::string(bpePiece).c_str());
      }
    }
    ret.insert(ret.end(), ids.begin(), ids.end());
    if (enableCache_) {
      cache_->put(token, std::move(ids));
    }
  }
  return ret;
}

std::vector<std::string_view> BPE::bpeV1(std::string_view text) {
  struct WordsRank {
    uint32_t pos;
    uint32_t rank;
  };

  auto words = ByteLevel::splitUTF8(text);
  std::vector<WordsRank> ranks(words.size() + 1);
  for (uint32_t i = 0; i < ranks.size(); i++) {
    ranks[i].pos = words[i].data() - text.data();
    ranks[i].rank = std::numeric_limits<uint32_t>::max();
  }
  ranks.back().pos = text.size();
  ranks.back().rank = std::numeric_limits<uint32_t>::max();

  auto getRank = [&text, &ranks, this](uint32_t idx1, uint32_t idx2, uint32_t idx3) -> uint32_t {
    if (idx3 >= ranks.size()) {
      return std::numeric_limits<uint32_t>::max();
    }
    const auto start = ranks[idx1].pos;
    const auto end1 = ranks[idx2].pos;
    const auto end2 = ranks[idx3].pos;
    std::string_view str1 = {text.data() + start, end1 - start};
    std::string_view str2 = {text.data() + end1, end2 - end1};
    const auto it = mergeRanks_.find({str1, str2});
    if (it != mergeRanks_.end()) {
      return it->second;
    }
    return std::numeric_limits<uint32_t>::max();
  };

  for (uint32_t i = 0; i < ranks.size() - 2; i++) {
    ranks[i].rank = getRank(i, i + 1, i + 2);
  }

  while (true) {
    if (ranks.size() == 1) {
      break;
    }

    auto minRank = std::make_pair<uint32_t, uint32_t>(std::numeric_limits<uint32_t>::max(), 0);
    for (uint32_t i = 0; i < ranks.size() - 1; i++) {
      auto rank = ranks[i].rank;
      if (rank < minRank.first) {
        minRank = {rank, i};
      }
    }

    if (minRank.first == std::numeric_limits<uint32_t>::max()) {
      break;
    }

    auto minIdx = minRank.second;
    ranks[minIdx].rank = getRank(minIdx, minIdx + 2, minIdx + 3);
    if (minIdx > 0) {
      ranks[minIdx - 1].rank = getRank(minIdx - 1, minIdx, minIdx + 2);
    }
    ranks.erase(ranks.begin() + (minIdx + 1));
  }

  std::vector<std::string_view> ret;
  ret.reserve(ranks.size() - 1);
  for (uint32_t i = 0; i < ranks.size() - 1; i++) {
    ret.emplace_back(text.data() + ranks[i].pos, ranks[i + 1].pos - ranks[i].pos);
  }
  return ret;
}

std::vector<std::string_view> BPE::bpeV2(std::string_view text) {
  struct WordsRank {
    uint32_t pos;
    uint32_t rank;
    WordsRank* next;
  };

  auto words = ByteLevel::splitUTF8(text);
  std::vector<WordsRank> ranks(words.size() + 1);

  // init ranks
  for (uint32_t i = 0; i < ranks.size(); i++) {
    ranks[i].pos = words[i].data() - text.data();
    ranks[i].rank = std::numeric_limits<uint32_t>::max();
    ranks[i].next = &ranks[i + 1];
  }
  ranks.back().pos = text.size();
  ranks.back().rank = std::numeric_limits<uint32_t>::max();
  ranks.back().next = nullptr;

  auto getRank = [&text, this](WordsRank* ptr) -> uint32_t {
    if (ptr->next->next == nullptr) {
      return std::numeric_limits<uint32_t>::max();
    }
    const auto start = ptr->pos;
    const auto end1 = ptr->next->pos;
    const auto end2 = ptr->next->next->pos;
    std::string_view str1 = {text.data() + start, end1 - start};
    std::string_view str2 = {text.data() + end1, end2 - end1};
    const auto it = mergeRanks_.find({str1, str2});
    if (it != mergeRanks_.end()) {
      return it->second;
    }
    return std::numeric_limits<uint32_t>::max();
  };

  // update ranks
  for (uint32_t i = 0; i < ranks.size() - 2; i++) {
    ranks[i].rank = getRank(&ranks[i]);
  }

  uint32_t validRanksCnt = 0;
  while (true) {
    auto minRank = std::numeric_limits<uint32_t>::max();
    WordsRank* minRankPtr = nullptr;
    WordsRank* minRankPrevPtr = nullptr;

    WordsRank* prev = nullptr;
    WordsRank* curr = &ranks[0];
    validRanksCnt = 0;
    while (curr) {
      if (curr->rank < minRank) {
        minRank = curr->rank;
        minRankPtr = curr;
        minRankPrevPtr = prev;
      }
      prev = curr;
      curr = curr->next;
      validRanksCnt++;
    }

    if (minRankPtr == nullptr) {
      break;
    }

    // merge with next
    minRankPtr->next = minRankPtr->next->next;
    minRankPtr->rank = getRank(minRankPtr);

    // update prev rank
    if (minRankPrevPtr != nullptr) {
      minRankPrevPtr->rank = getRank(minRankPrevPtr);
    }
  }

  std::vector<std::string_view> ret;
  ret.reserve(validRanksCnt - 1);
  for (auto* ptr = &ranks[0]; ptr && ptr->next; ptr = ptr->next) {
    ret.emplace_back(text.data() + ptr->pos, ptr->next->pos - ptr->pos);
  }
  return ret;
}

}  // namespace tinygpt::tokenizer