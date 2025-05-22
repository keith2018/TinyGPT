/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "BPE.h"

#include <limits>
#include <queue>

#include "ByteLevel.h"

namespace tinygpt::tokenizer {

LRUCache::LRUCache(size_t capacity) : capacity_(capacity) { assert(capacity > 0); }

std::optional<LRUCache::Value> LRUCache::get(const Key& key) const {
  const auto it = map_.find(key);
  if (it == map_.end()) {
    return std::nullopt;
  }
  list_.splice(list_.begin(), list_, it->second.second);
  return it->second.first;
}

void LRUCache::put(const Key& key, Value&& value) const {
  const auto it = map_.find(key);
  if (it != map_.end()) {
    it->second.first = std::move(value);
    list_.splice(list_.begin(), list_, it->second.second);
  } else {
    if (map_.size() == capacity_) {
      const Key& old_key = list_.back();
      map_.erase(old_key);
      list_.pop_back();
    }
    list_.push_front(key);
    map_.emplace(key, std::make_pair(std::move(value), list_.begin()));
  }
}

void LRUCache::erase(const Key& key) const {
  const auto it = map_.find(key);
  if (it != map_.end()) {
    list_.erase(it->second.second);
    map_.erase(it);
  }
}

size_t LRUCache::size() const { return map_.size(); }

BPE::BPE(const ankerl::unordered_dense::map<std::string, int32_t>& vocab,
         const ankerl::unordered_dense::map<StringPair, int32_t, StringPairHash>& merges, bool ignoreMerges,
         bool enableCache)
    : ignoreMerges_(ignoreMerges), enableCache_(enableCache) {
  // encoder_ & decoder_
  size_t encoderStrLen = 0;
  int32_t maxTokenId = -std::numeric_limits<int32_t>::max();
  for (auto& [k, v] : vocab) {
    encoderStrLen += k.size();
    maxTokenId = std::max(maxTokenId, v);
  }
  encoderBackStr_.reserve(encoderStrLen);
  encoder_.reserve(vocab.size());
  decoder_.resize(maxTokenId + 1);
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
  if (id >= 0 && id < static_cast<int32_t>(decoder_.size())) {
    return decoder_[id];
  }

  LOGE("error decode id: %d", id);
  return {};
}

std::vector<int32_t> BPE::tokenize(const StringPieces& tokens) {
  thread_local LRUCache cache_;

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
      auto cacheItem = cache_.get(token);
      if (cacheItem) {
        ret.insert(ret.end(), cacheItem.value().begin(), cacheItem.value().end());
        continue;
      }
    }

    // bpe
    constexpr size_t FAST_BPE_THRESHOLD = 32;
    auto bpePieces = token.size() > FAST_BPE_THRESHOLD ? bpeV2(token) : bpeV1(token);
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
      cache_.put(token, std::move(ids));
    }
  }
  return ret;
}

std::vector<std::string_view> BPE::bpeV1(std::string_view text) {
  struct Node {
    uint32_t pos;
    uint32_t rank;
  };

  auto words = ByteLevel::splitUTF8(text);
  std::vector<Node> nodes(words.size() + 1);
  for (uint32_t i = 0; i < nodes.size(); i++) {
    nodes[i].pos = (i < nodes.size() - 1) ? (words[i].data() - text.data()) : text.size();
    nodes[i].rank = std::numeric_limits<uint32_t>::max();
  }

  auto getRank = [&text, &nodes, this](uint32_t idx1, uint32_t idx2, uint32_t idx3) -> uint32_t {
    if (idx3 >= nodes.size()) {
      return std::numeric_limits<uint32_t>::max();
    }
    const auto start = nodes[idx1].pos;
    const auto end1 = nodes[idx2].pos;
    const auto end2 = nodes[idx3].pos;
    std::string_view str1 = {text.data() + start, end1 - start};
    std::string_view str2 = {text.data() + end1, end2 - end1};
    const auto it = mergeRanks_.find({str1, str2});
    if (it != mergeRanks_.end()) {
      return it->second;
    }
    return std::numeric_limits<uint32_t>::max();
  };

  for (uint32_t i = 0; i < nodes.size() - 2; i++) {
    nodes[i].rank = getRank(i, i + 1, i + 2);
  }

  while (true) {
    if (nodes.size() == 1) {
      break;
    }

    auto minRank = std::make_pair<uint32_t, uint32_t>(std::numeric_limits<uint32_t>::max(), 0);
    for (uint32_t i = 0; i < nodes.size() - 1; i++) {
      auto rank = nodes[i].rank;
      if (rank < minRank.first) {
        minRank = {rank, i};
      }
    }

    if (minRank.first == std::numeric_limits<uint32_t>::max()) {
      break;
    }

    auto minIdx = minRank.second;
    nodes[minIdx].rank = getRank(minIdx, minIdx + 2, minIdx + 3);
    if (minIdx > 0) {
      nodes[minIdx - 1].rank = getRank(minIdx - 1, minIdx, minIdx + 2);
    }
    nodes.erase(nodes.begin() + (minIdx + 1));
  }

  std::vector<std::string_view> ret;
  ret.reserve(nodes.size() - 1);
  for (uint32_t i = 0; i < nodes.size() - 1; i++) {
    ret.emplace_back(text.data() + nodes[i].pos, nodes[i + 1].pos - nodes[i].pos);
  }
  return ret;
}

std::vector<std::string_view> BPE::bpeV2(std::string_view text) {
  struct Node {
    uint32_t pos;
    uint32_t rank;
    Node* prev;
    Node* next;
    bool active;
  };

  auto words = ByteLevel::splitUTF8(text);
  std::vector<Node> nodes(words.size() + 1);

  for (uint32_t i = 0; i < nodes.size(); i++) {
    nodes[i].pos = (i < words.size()) ? (words[i].data() - text.data()) : text.size();
    nodes[i].prev = (i > 0) ? &nodes[i - 1] : nullptr;
    nodes[i].next = (i < nodes.size() - 1) ? &nodes[i + 1] : nullptr;
    nodes[i].rank = std::numeric_limits<uint32_t>::max();
    nodes[i].active = true;
  }

  auto getRank = [&text, this](const Node* node) -> uint32_t {
    if (!node || !node->next || !node->next->next) {
      return std::numeric_limits<uint32_t>::max();
    }

    const uint32_t start = node->pos;
    const uint32_t mid = node->next->pos;
    const uint32_t end = node->next->next->pos;

    std::string_view str1(text.data() + start, mid - start);
    std::string_view str2(text.data() + mid, end - mid);

    auto it = mergeRanks_.find({str1, str2});
    if (it != mergeRanks_.end()) {
      return it->second;
    }
    return std::numeric_limits<uint32_t>::max();
  };

  using QueueElem = std::pair<uint32_t, Node*>;  // <rank, node>
  auto cmp = [](const QueueElem& a, const QueueElem& b) {
    // If ranks are equal, prioritize the smaller index
    if (a.first == b.first) {
      return a.second->pos > b.second->pos;
    }
    return a.first > b.first;
  };
  std::priority_queue<QueueElem, std::vector<QueueElem>, decltype(cmp)> pq(cmp);
  for (auto& node : nodes) {
    if (node.next) {
      node.rank = getRank(&node);
      if (node.rank != std::numeric_limits<uint32_t>::max()) {
        pq.emplace(node.rank, &node);
      }
    }
  }

  auto activeCount = nodes.size();
  while (!pq.empty()) {
    auto [minRank, node] = pq.top();
    pq.pop();

    if (!node->active || node->rank != minRank) {
      continue;
    }
    Node* nextNode = node->next;
    if (!nextNode || !nextNode->active) {
      continue;
    }

    // mark not active
    nextNode->active = false;
    activeCount--;

    // merge with next
    node->next = nextNode->next;
    if (node->next) {
      node->next->prev = node;
    }

    // update rank
    const uint32_t oldRank = node->rank;
    node->rank = getRank(node);
    if (node->rank != oldRank && node->rank != std::numeric_limits<uint32_t>::max()) {
      pq.emplace(node->rank, node);
    }

    // update prev node
    if (node->prev && node->prev->active) {
      const uint32_t prevOldRank = node->prev->rank;
      node->prev->rank = getRank(node->prev);
      if (node->prev->rank != prevOldRank && node->prev->rank != std::numeric_limits<uint32_t>::max()) {
        pq.emplace(node->prev->rank, node->prev);
      }
    }
  }

  std::vector<std::string_view> ret;
  ret.reserve(activeCount - 1);
  for (Node* curr = &nodes[0]; curr; curr = curr->next) {
    if (curr->active && curr->next) {
      ret.emplace_back(text.data() + curr->pos, curr->next->pos - curr->pos);
    }
  }

  return ret;
}

}  // namespace tinygpt::tokenizer