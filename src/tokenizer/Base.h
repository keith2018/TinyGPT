/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ankerl/unordered_dense.h"
#include "re2/re2.h"

#include <TinyTorch/Logger.h>

namespace tinygpt::tokenizer {

constexpr uint32_t NUM_MAX_THREAD = 128;

enum class ComponentType {
  UNKNOWN = 0,
  SEQUENCE,
  SPLIT,
  BYTE_LEVEL,
  BPE,
};

enum class SplitDelimiterBehavior {
  UNKNOWN = 0,
  REMOVED,
  ISOLATED,
  MERGED_WITH_PREVIOUS,
  MERGED_WITH_NEXT,
  CONTIGUOUS
};

using Range = std::pair<uint32_t, uint32_t>;  // [start, stop]
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

struct PreTokenizedString {
  std::vector<Range> pieces;
  std::string backStr;
};

class Component {
 public:
  virtual ~Component() = default;

  virtual ComponentType getType() = 0;

  // Normalizer
  virtual std::string normalize(std::string_view text) { return {}; }

  // PreTokenizer
  virtual PreTokenizedString preTokenize(std::string_view text) { return {}; }

  // Model
  virtual int32_t token2Id(const std::string &token) { return -1; }
  virtual std::string id2Token(int32_t id) { return {}; }
  virtual std::vector<int32_t> tokenize(const PreTokenizedString &tokens) { return {}; }

  // PostProcessor
  virtual std::vector<int32_t> postProcess(const std::vector<int32_t> &ids) { return {}; }

  // Decoder
  virtual std::string decode(const std::vector<std::string> &pieces) { return {}; }
};

class ComponentSequence : public Component {
 public:
  ComponentType getType() override { return ComponentType::SEQUENCE; }

  void addComponent(std::unique_ptr<Component> &&component);

  // PreTokenizer
  PreTokenizedString preTokenize(std::string_view text) override;

  // PostProcessor
  std::vector<int32_t> postProcess(const std::vector<int32_t> &ids) override;

 protected:
  std::vector<std::unique_ptr<Component>> components;
};

}  // namespace tinygpt::tokenizer
