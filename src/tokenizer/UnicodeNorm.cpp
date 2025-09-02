/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "UnicodeNorm.h"

#include "utf8proc/utf8proc.h"

namespace tinygpt::tokenizer {

inline std::string normalizeImpl(std::string_view text, utf8proc_option_t options) {
  utf8proc_uint8_t* result = nullptr;
  utf8proc_ssize_t len =
      utf8proc_map(reinterpret_cast<const utf8proc_uint8_t*>(text.data()), static_cast<utf8proc_ssize_t>(text.size()),
                   &result, static_cast<utf8proc_option_t>(options | UTF8PROC_STABLE));

  if (len < 0) {
    auto* errMsg = utf8proc_errmsg(len);
    LOGE("utf8proc error: %s", errMsg);
    return std::string(text);
  }

  std::string normalized(reinterpret_cast<char*>(result), static_cast<size_t>(len));
  free(result);
  return normalized;
}

std::string NFC::normalize(std::string_view text) { return normalizeImpl(text, UTF8PROC_COMPOSE); }

std::string NFD::normalize(std::string_view text) { return normalizeImpl(text, UTF8PROC_DECOMPOSE); }

std::string NFKC::normalize(std::string_view text) {
  return normalizeImpl(text, static_cast<utf8proc_option_t>(UTF8PROC_COMPOSE | UTF8PROC_COMPAT));
}

std::string NFKD::normalize(std::string_view text) {
  return normalizeImpl(text, static_cast<utf8proc_option_t>(UTF8PROC_DECOMPOSE | UTF8PROC_COMPAT));
}

}  // namespace tinygpt::tokenizer