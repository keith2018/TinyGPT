/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "ByteLevel.h"

#include "Split.h"

namespace tinygpt::tokenizer {

// Ref https://github.com/openai/gpt-2/blob/master/src/encoder.py
std::vector<char32_t> ByteLevel::bytesToUtf8() {
  std::vector<char32_t> b2u(256, 0);

  std::vector<std::pair<uint8_t, uint8_t>> ranges = {{'!', '~'}, {'\xA1', '\xAC'}, {'\xAE', '\xFF'}};
  for (auto& [start, end] : ranges) {
    for (int16_t i = start; i <= end; i++) {
      b2u[i] = static_cast<char32_t>(i);
    }
  }

  uint8_t n = 0;
  for (int16_t i = 0; i < 256; i++) {
    if (b2u[i] == 0) {
      b2u[i] = static_cast<char32_t>(256 + n);
      n++;
    }
  }
  return b2u;
}

static std::array<std::array<char, 2>, 256> makeByteUtf8Table(const std::vector<char32_t>& bytesChar,
                                                              std::array<uint8_t, 256>& outLen) {
  std::array<std::array<char, 2>, 256> table{};
  for (int16_t i = 0; i < 256; i++) {
    char32_t cp = bytesChar[i];
    if (cp <= 0x7F) {
      table[i][0] = static_cast<char>(cp);
      outLen[i] = 1;
    } else if (cp <= 0x7FF) {
      table[i][0] = static_cast<char>(0xC0 | (cp >> 6));
      table[i][1] = static_cast<char>(0x80 | (cp & 0x3F));
      outLen[i] = 2;
    } else {
      // invalid
      table[i][0] = '?';
      outLen[i] = 1;
    }
  }
  return table;
}

const std::vector<char32_t> ByteLevel::bytesChar_ = bytesToUtf8();
const std::array<uint8_t, 256> ByteLevel::byteUtf8Len_ = [] {
  std::array<uint8_t, 256> len{};
  makeByteUtf8Table(ByteLevel::bytesChar_, len);
  return len;
}();
const std::array<std::array<char, 2>, 256> ByteLevel::byteUtf8Table_ = [] {
  std::array<uint8_t, 256> len{};
  return makeByteUtf8Table(ByteLevel::bytesChar_, len);
}();

std::string ByteLevel::utf8ToBytes(const std::string& str) {
  static std::unordered_map<std::string, uint8_t> utf8ToByte;
  static bool inited = false;
  if (!inited) {
    for (int16_t i = 0; i < 256; ++i) {
      const auto len = byteUtf8Len_[i];
      std::string key(byteUtf8Table_[i].data(), len);
      utf8ToByte[key] = static_cast<uint8_t>(i);
    }
    inited = true;
  }

  std::string result;
  size_t i = 0;
  while (i < str.size()) {
    bool found = false;
    for (auto len = 1; len <= 2; len++) {
      if (i + len <= str.size()) {
        std::string key = str.substr(i, len);
        auto it = utf8ToByte.find(key);
        if (it != utf8ToByte.end()) {
          result.push_back(static_cast<char>(it->second));
          i += len;
          found = true;
          break;
        }
      }
    }
    if (!found) {
      LOGE("Invalid byte level utf8 string %s at position %d", str.c_str(), i);
    }
  }
  return result;
}

ByteLevel::ByteLevel(bool addPrefixSpace, bool useRegex) : addPrefixSpace_(addPrefixSpace), useRegex_(useRegex) {
  if (useRegex_) {
    // Ref https://github.com/openai/gpt-2/blob/master/src/encoder.py
    static const std::string PATTERN_GPT2 =
        R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";
    matcher_ = std::make_unique<Regex>(PATTERN_GPT2);
    assert(matcher_->valid());
  }
}

PreTokenizedString ByteLevel::preTokenize(std::string_view text) {
  std::string_view inputView;
  std::string inputStr;
  if (addPrefixSpace_ && !text.empty() && text[0] != ' ') {
    inputStr = ' ';
    inputStr += text;
    inputView = inputStr;
  } else {
    inputView = text;
  }

  std::vector<Range> ranges;
  if (useRegex_) {
    ranges = Split::split(inputView, *matcher_, SplitDelimiterBehavior::ISOLATED);
  } else {
    ranges = {{0, inputView.size()}};
  }

  PreTokenizedString ret;
  ret.pieces.reserve(ranges.size());

  size_t strLen = 0;
  for (const auto& r : ranges) {
    strLen += r.second - r.first;
  }
  ret.backStr.reserve(strLen * 2);

  for (const auto& r : ranges) {
    const auto pos = ret.backStr.size();
    for (uint32_t i = r.first; i < r.second; i++) {
      const auto ch = static_cast<uint8_t>(inputView[i]);
      ret.backStr.append(byteUtf8Table_[ch].data(), byteUtf8Len_[ch]);
    }
    ret.pieces.emplace_back(pos, ret.backStr.size());
  }
  return ret;
}

std::vector<int32_t> ByteLevel::postProcess(const std::vector<int32_t>& ids, bool addSpecialTokens) {
  // 'trimOffsets' not support
  return ids;
}

std::string ByteLevel::decode(const std::vector<std::string>& pieces) {
  std::string ret;
  size_t len = 0;
  for (auto& s : pieces) {
    len += s.size();
  }
  ret.reserve(len);
  for (auto& s : pieces) {
    ret.append(s);
  }
  return ret;
}

}  // namespace tinygpt::tokenizer