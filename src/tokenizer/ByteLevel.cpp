/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "ByteLevel.h"

#include "Split.h"
#include "utf8proc/utf8proc.h"

namespace tinygpt::tokenizer {

class ByteLevelHelper {
 public:
  static const ByteLevelHelper& instance() {
    static const ByteLevelHelper helper;
    return helper;
  }
  const std::array<char32_t, 256>& getBytesChar() const { return bytesChar_; }
  const std::array<std::array<char, 4>, 256>& getTable() const { return table_; }
  const std::array<uint8_t, 256>& getLengths() const { return lengths_; }

 private:
  ByteLevelHelper() {
    // Ref https://github.com/openai/gpt-2/blob/master/src/encoder.py
    bytesChar_.fill(0);
    std::vector<std::pair<uint8_t, uint8_t>> ranges = {{'!', '~'}, {'\xA1', '\xAC'}, {'\xAE', '\xFF'}};
    for (auto& [start, end] : ranges) {
      for (int16_t i = start; i <= end; i++) {
        bytesChar_[i] = static_cast<char32_t>(i);
      }
    }
    uint8_t n = 0;
    for (int16_t i = 0; i < 256; i++) {
      if (bytesChar_[i] == 0) {
        bytesChar_[i] = static_cast<char32_t>(256 + n);
        n++;
      }
    }

    for (int16_t i = 0; i < 256; i++) {
      uint8_t buf[4];
      auto len = utf8proc_encode_char(static_cast<utf8proc_int32_t>(bytesChar_[i]), buf);
      if (len < 0) {
        table_[i][0] = '?';
        lengths_[i] = 1;
      } else {
        for (int j = 0; j < len; j++) {
          table_[i][j] = static_cast<char>(buf[j]);
        }
        lengths_[i] = static_cast<uint8_t>(len);
      }
    }
  }
  std::array<char32_t, 256> bytesChar_{};
  std::array<std::array<char, 4>, 256> table_{};
  std::array<uint8_t, 256> lengths_{};
};

const std::array<char32_t, 256> ByteLevel::bytesChar_ = [] { return ByteLevelHelper::instance().getBytesChar(); }();

const std::array<uint8_t, 256> ByteLevel::byteUtf8Len_ = [] { return ByteLevelHelper::instance().getLengths(); }();

const std::array<std::array<char, 4>, 256> ByteLevel::byteUtf8Table_ = [] {
  return ByteLevelHelper::instance().getTable();
}();

const std::array<uint8_t, UNICODE_CODEPOINT_MAX> ByteLevel::codepointByteTable_ = [] {
  static std::array<uint8_t, UNICODE_CODEPOINT_MAX> table{};
  table.fill(0);  // default: 0
  for (int i = 0; i < 256; i++) {
    auto cp = static_cast<uint32_t>(bytesChar_[i]);
    table[cp] = static_cast<uint8_t>(i);
  }
  return table;
}();

std::string ByteLevel::utf8ToBytes(std::string_view str) {
  std::string result;
  result.reserve(str.size());

  const auto* data = reinterpret_cast<const uint8_t*>(str.data());
  auto len = static_cast<utf8proc_ssize_t>(str.size());
  utf8proc_ssize_t i = 0;

  while (i < len) {
    utf8proc_int32_t codepoint;
    auto charLen = utf8proc_iterate(data + i, len - i, &codepoint);
    if (charLen < 0) {
      LOGE("ByteLevel: invalid UTF-8 at position %zu", i);
      ASSERT(false);
      break;
    }
    if (codepoint < 0 || codepoint >= static_cast<int>(codepointByteTable_.size())) {
      LOGE("ByteLevel: codepoint %d out of range", codepoint);
      ASSERT(false);
      break;
    }
    auto byte = codepointByteTable_[codepoint];
    if (byte == 0) {
      result.append(std::string_view{str.data() + i, static_cast<size_t>(charLen)});
    } else {
      result.push_back(static_cast<char>(byte));
    }
    i += charLen;
  }
  return result;
}

int32_t ByteLevel::findIncompletePos(std::string_view str) {
  const auto len = static_cast<int32_t>(str.size());
  if (len == 0) {
    return -1;
  }

  const auto* data = reinterpret_cast<const uint8_t*>(str.data());
  int32_t start = std::max(0, len - 4);
  for (int32_t pos = start; pos < len; pos++) {
    utf8proc_int32_t codepoint;
    auto charLen = utf8proc_iterate(data + pos, len - pos, &codepoint);
    if (charLen < 0) {
      return pos;
    }
    if (pos + charLen > len) {
      return pos;
    }
  }
  return -1;
}

std::vector<std::string_view> ByteLevel::splitUTF8(std::string_view str) {
  std::vector<std::string_view> results;
  results.reserve((str.size() + 1) / 2);

  const auto* data = reinterpret_cast<const uint8_t*>(str.data());
  auto len = static_cast<utf8proc_ssize_t>(str.size());
  utf8proc_ssize_t i = 0;

  while (i < len) {
    utf8proc_int32_t codepoint;
    auto charLen = utf8proc_iterate(data + i, len - i, &codepoint);
    if (charLen < 0) {
      LOGE("splitUTF8 error: invalid UTF-8 at pos %zu", i);
      charLen = 1;  // skip invalid byte
    }
    results.emplace_back(str.data() + i, static_cast<size_t>(charLen));
    i += charLen;
  }
  return results;
}

ByteLevel::ByteLevel(bool addPrefixSpace, bool useRegex) : addPrefixSpace_(addPrefixSpace), useRegex_(useRegex) {
  if (useRegex_) {
    // Ref https://github.com/openai/gpt-2/blob/master/src/encoder.py
    static const std::string PATTERN_GPT2 =
        R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";
    matcher_ = std::make_unique<Regex>(PATTERN_GPT2);
    ASSERT(matcher_->valid());
  }
}

StringPieces ByteLevel::preTokenize(const StringPieces& text) {
  ASSERT(!text.pieces.empty());
  ASSERT(!text.backStr.empty());

  if (!useRegex_) {
    auto& pieces = text.pieces;
    auto& firstRange = pieces[0];
    if (!addPrefixSpace_ || text.backStr[firstRange.first] == ' ') {
      return byteLevelEncode(&pieces[0], pieces.size(), text.backStr, {});
    } else {
      std::string firstPiece;
      firstPiece.reserve(firstRange.second - firstRange.first + 1);
      firstPiece.push_back(' ');
      firstPiece.append(text.backStr.data() + firstRange.first, firstRange.second - firstRange.first);
      return byteLevelEncode(pieces.size() > 1 ? &pieces[1] : nullptr, pieces.size() - 1, text.backStr, firstPiece);
    }
  } else {
    std::string_view inputView = text.backStr;
    std::string inputWithSpace;
    if (addPrefixSpace_ && inputView[0] != ' ') {
      inputWithSpace.reserve(inputView.size() + 1);
      inputWithSpace.push_back(' ');
      inputWithSpace.append(inputView);
      inputView = inputWithSpace;
    }
    auto pieces = Split::split(inputView, *matcher_, SplitDelimiterBehavior::ISOLATED);
    return byteLevelEncode(&pieces[0], pieces.size(), inputView, {});
  }
}

StringPieces ByteLevel::byteLevelEncode(const Range* pieces, size_t pieceCnt, std::string_view backStr,
                                        std::string_view firstPiece) {
  StringPieces ret;
  ret.pieces.reserve(pieceCnt + (firstPiece.empty() ? 0 : 1));
  ret.backStr.reserve(backStr.size() * 2);

  auto appendPiece = [&](std::string_view sv) {
    const auto pos = ret.backStr.size();
    for (auto c : sv) {
      const auto ch = static_cast<uint8_t>(c);
      ret.backStr.append(byteUtf8Table_[ch].data(), byteUtf8Len_[ch]);
    }
    ret.pieces.emplace_back(pos, ret.backStr.size());
  };

  if (!firstPiece.empty()) {
    appendPiece(firstPiece);
  }

  for (size_t i = 0; i < pieceCnt; i++) {
    auto& r = pieces[i];
    appendPiece(backStr.substr(r.first, r.second - r.first));
  }
  return ret;
}

std::vector<int32_t> ByteLevel::postProcess(const std::vector<int32_t>& ids, bool addSpecialTokens) {
  // 'trimOffsets' not support
  return ids;
}

std::vector<std::string> ByteLevel::decode(const std::vector<std::string>& pieces) { return pieces; }

}  // namespace tinygpt::tokenizer