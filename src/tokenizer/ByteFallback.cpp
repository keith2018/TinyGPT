/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "ByteFallback.h"

#include "utf8proc/utf8proc.h"

namespace tinygpt::tokenizer {

// Ref: https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/decoders/byte_fallback.rs
std::vector<std::string> ByteFallback::decode(const std::vector<std::string> &pieces) {
  std::vector<std::string> retPieces;
  retPieces.reserve(pieces.size());

  std::vector<uint8_t> previousBytes;
  auto processByteTokens = [&]() {
    if (previousBytes.empty()) {
      return;
    }

    if (isValidUtf8(previousBytes)) {
      retPieces.emplace_back(reinterpret_cast<const char *>(previousBytes.data()), previousBytes.size());
    } else {
      for (size_t i = 0; i < previousBytes.size(); i++) {
        retPieces.emplace_back("\ufffd");
      }
    }
    previousBytes.clear();
  };

  for (const auto &token : pieces) {
    if (token.size() == 6 && token.substr(0, 3) == "<0x" && token[5] == '>' && std::isxdigit(token[3]) &&
        std::isxdigit(token[4])) {
      char *endPtr = nullptr;
      std::string hexPart = token.substr(3, 2);
      unsigned long value = std::strtoul(hexPart.c_str(), &endPtr, 16);

      if (*endPtr == '\0' && value <= 255) {
        previousBytes.push_back(static_cast<uint8_t>(value));
      } else {
        processByteTokens();
        retPieces.push_back(token);
      }
    } else {
      processByteTokens();
      retPieces.push_back(token);
    }
  }
  processByteTokens();
  return retPieces;
}

bool ByteFallback::isValidUtf8(const std::vector<uint8_t> &bytes) {
  if (bytes.empty()) {
    return true;
  }
  const uint8_t *data = bytes.data();
  const auto len = static_cast<utf8proc_ssize_t>(bytes.size());
  for (utf8proc_ssize_t i = 0; i < len;) {
    utf8proc_int32_t codepoint;
    utf8proc_ssize_t n = utf8proc_iterate(data + i, len - i, &codepoint);
    if (n < 0 || codepoint == -1) {
      return false;
    }
    i += n;
  }
  return true;
}

}  // namespace tinygpt::tokenizer