/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <vector>

#include "Base.h"
#include "Regex.h"

namespace tinygpt::tokenizer {

static constexpr size_t UNICODE_CODEPOINT_MAX = 0x110000;

class ByteLevel : public Component {
 public:
  explicit ByteLevel(bool addPrefixSpace = false, bool useRegex = false);

  ComponentType getType() override { return ComponentType::BYTE_LEVEL; }

  StringPieces preTokenize(const StringPieces &text) override;
  std::vector<int32_t> postProcess(const std::vector<int32_t> &ids, bool addSpecialTokens) override;
  std::vector<std::string> decode(const std::vector<std::string> &pieces) override;

  static const std::array<char32_t, 256> &alphabet() { return bytesChar_; }
  static std::string utf8ToBytes(std::string_view str);
  static int32_t findIncompletePos(std::string_view str);
  static std::vector<std::string_view> splitUTF8(std::string_view str);

 private:
  static StringPieces byteLevelEncode(const Range *pieces, size_t pieceCnt, std::string_view backStr,
                                      std::string_view firstPiece);
  bool addPrefixSpace_;
  bool useRegex_;

  std::unique_ptr<Regex> matcher_;

  static const std::array<char32_t, 256> bytesChar_;
  static const std::array<uint8_t, 256> byteUtf8Len_;
  static const std::array<std::array<char, 4>, 256> byteUtf8Table_;
  static const std::array<uint8_t, UNICODE_CODEPOINT_MAX> codepointByteTable_;
};

}  // namespace tinygpt::tokenizer
