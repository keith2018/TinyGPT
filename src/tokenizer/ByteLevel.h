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

class ByteLevel : public Component {
 public:
  explicit ByteLevel(bool addPrefixSpace = false, bool useRegex = false);

  ComponentType getType() override { return ComponentType::BYTE_LEVEL; }

  StringPieces preTokenize(const StringPieces &text) override;
  std::vector<int32_t> postProcess(const std::vector<int32_t> &ids, bool addSpecialTokens) override;
  std::string decode(const std::vector<std::string> &pieces) override;

  static const std::vector<char32_t> &alphabet() { return bytesChar_; }
  static std::string utf8ToBytes(const std::string &str);
  static std::vector<std::string_view> splitUTF8(std::string_view str);

 private:
  static StringPieces byteLevelEncode(const Range *pieces, size_t pieceCnt, std::string_view backStr,
                                      std::string_view firstPiece);
  static std::vector<char32_t> bytesToUtf8();

  bool addPrefixSpace_;
  bool useRegex_;

  std::unique_ptr<Regex> matcher_;
  static const std::vector<char32_t> bytesChar_;
  static const std::array<std::array<char, 2>, 256> byteUtf8Table_;
  static const std::array<uint8_t, 256> byteUtf8Len_;
};

}  // namespace tinygpt::tokenizer
