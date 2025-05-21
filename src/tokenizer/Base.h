/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <cassert>
#include <memory>
#include <string>
#include <vector>

#include "TinyTorch/Logger.h"
#include "ankerl/unordered_dense.h"

namespace tinygpt::tokenizer {

enum class ComponentType {
  UNKNOWN = 0,
  SEQUENCE,
  SPLIT,
  BYTE_LEVEL,
  BPE,
  TEMPLATE_PROCESSING,
};

using Range = std::pair<uint32_t, uint32_t>;  // [begin, end]
using StringPair = std::pair<std::string, std::string>;
using StringViewPair = std::pair<std::string_view, std::string_view>;

struct StringPairHash {
  using is_avalanching = void;  // mark class as high quality avalanching hash

  size_t operator()(const StringPair &p) const {
    ankerl::unordered_dense::hash<std::string> hasher;
    return combine(hasher(p.first), hasher(p.second));
  }

  static size_t combine(size_t h1, size_t h2) noexcept { return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2)); }
};

struct StringViewPairHash {
  using is_avalanching = void;  // mark class as high quality avalanching hash

  size_t operator()(const StringViewPair &p) const {
    ankerl::unordered_dense::hash<std::string_view> hasher;
    return combine(hasher(p.first), hasher(p.second));
  }

  static size_t combine(size_t h1, size_t h2) noexcept { return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2)); }
};

struct StringPieces {
  std::vector<Range> pieces;
  std::string backStr;

  StringPieces() = default;

  StringPieces(const char *str) {  // NOLINT
    backStr = str;
    pieces = {{0, backStr.size()}};
  }

  StringPieces(std::string_view str) {  // NOLINT
    backStr = str;
    pieces = {{0, backStr.size()}};
  }

  StringPieces(std::string &&str) {  // NOLINT
    backStr = std::move(str);
    pieces = {{0, backStr.size()}};
  }
};

class Component {
 public:
  virtual ~Component() = default;

  virtual ComponentType getType() = 0;

  // Normalizer
  virtual std::string normalize(std::string_view text) { return {}; }

  // PreTokenizer
  virtual StringPieces preTokenize(const StringPieces &text) { return {}; }

  // Model
  virtual int32_t token2Id(const std::string &token) { return -1; }
  virtual std::string id2Token(int32_t id) { return {}; }
  virtual std::vector<int32_t> tokenize(const StringPieces &tokens) { return {}; }

  // PostProcessor
  virtual std::vector<int32_t> postProcess(const std::vector<int32_t> &ids, bool addSpecialTokens) { return {}; }

  // Decoder
  virtual std::string decode(const std::vector<std::string> &pieces) { return {}; }
};

class ComponentSequence : public Component {
 public:
  ComponentType getType() override { return ComponentType::SEQUENCE; }

  void addComponent(std::unique_ptr<Component> &&component);

  // PreTokenizer
  StringPieces preTokenize(const StringPieces &text) override;

  // PostProcessor
  std::vector<int32_t> postProcess(const std::vector<int32_t> &ids, bool addSpecialTokens) override;

 protected:
  std::vector<std::unique_ptr<Component>> components;
};

}  // namespace tinygpt::tokenizer
