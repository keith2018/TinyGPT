/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "Regex.h"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

namespace tinygpt::tokenizer {

class Regex::Impl {
 public:
  explicit Impl(std::string_view pattern);
  ~Impl();

  Impl(const Impl &) = delete;
  Impl &operator=(const Impl &) = delete;

  bool valid() const;
  void matchAll(std::vector<Range> &ret, std::string_view str) const;
  static std::string quoteMeta(std::string_view unquoted);

 private:
  static void printError(int error);

  bool jitEnabled_ = false;
  pcre2_code_8 *regex_ = nullptr;
};

Regex::Impl::Impl(std::string_view pattern) {
  int jit = 0;
  pcre2_config_8(PCRE2_CONFIG_JIT, &jit);
  jitEnabled_ = (jit != 0);

  int error = 0;
  PCRE2_SIZE errorOffset = 0;
  constexpr auto flags = PCRE2_UCP | PCRE2_UTF;
  regex_ = pcre2_compile_8(reinterpret_cast<PCRE2_SPTR8>(pattern.data()), pattern.size(), flags, &error, &errorOffset,
                           nullptr);
  if (!regex_) {
    printError(error);
    return;
  }

  // jit compile
  if (jitEnabled_) {
    error = pcre2_jit_compile_8(regex_, PCRE2_JIT_COMPLETE);
    if (error != 0) {
      printError(error);
      jitEnabled_ = false;
    }
  }
}

Regex::Impl::~Impl() {
  if (regex_) {
    pcre2_code_free_8(regex_);
    regex_ = nullptr;
  }
}

bool Regex::Impl::valid() const { return regex_ != nullptr; }

void Regex::Impl::matchAll(std::vector<Range> &ret, std::string_view str) const {
  if (str.empty() || !regex_) {
    return;
  }

  // thread_local cache
  thread_local pcre2_match_data_8 *matchData = nullptr;
  thread_local pcre2_code_8 *lastRegex = nullptr;

  if (matchData == nullptr || lastRegex != regex_) {
    if (matchData != nullptr) {
      pcre2_match_data_free_8(matchData);
    }
    matchData = pcre2_match_data_create_from_pattern_8(regex_, nullptr);
    lastRegex = regex_;
    if (!matchData) {
      LOGE("call pcre2_match_data_create_from_pattern_8 error: nullptr");
      return;
    }
  }

  const auto matcher = jitEnabled_ ? pcre2_jit_match_8 : pcre2_match_8;

  size_t offset = 0;
  while (offset < str.size()) {
    int rc = matcher(regex_, reinterpret_cast<PCRE2_SPTR8>(str.data()), str.size(), offset, 0, matchData, nullptr);
    if (rc < 0) {
      break;
    }

    const auto *ov = pcre2_get_ovector_pointer_8(matchData);
    const auto matchStart = ov[0], matchEnd = ov[1];
    if (matchEnd <= matchStart) {
      break;
    }
    ret.emplace_back(matchStart, matchEnd);
    offset = matchEnd;
  }
}

// Ref: https://github.com/google/re2/blob/main/re2/re2.cc
std::string Regex::Impl::quoteMeta(std::string_view unquoted) {
  std::string result;
  result.reserve(unquoted.size() << 1);

  // Escape any ascii character not in [A-Za-z_0-9].
  //
  // Note that it's legal to escape a character even if it has no
  // special meaning in a regular expression -- so this function does
  // that.  (This also makes it identical to the perl function of the
  // same name except for the null-character special case;
  // see `perldoc -f quotemeta`.)
  for (size_t ii = 0; ii < unquoted.size(); ++ii) {
    // Note that using 'isalnum' here raises the benchmark time from
    // 32ns to 58ns:
    if ((unquoted[ii] < 'a' || unquoted[ii] > 'z') && (unquoted[ii] < 'A' || unquoted[ii] > 'Z') &&
        (unquoted[ii] < '0' || unquoted[ii] > '9') && unquoted[ii] != '_' &&
        // If this is the part of a UTF8 or Latin1 character, we need
        // to copy this byte without escaping.  Experimentally this is
        // what works correctly with the regexp library.
        !(unquoted[ii] & 128)) {
      if (unquoted[ii] == '\0') {  // Special handling for null chars.
        // Note that this special handling is not strictly required for RE2,
        // but this quoting is required for other regexp libraries such as
        // PCRE.
        // Can't use "\\0" since the next character might be a digit.
        result += "\\x00";
        continue;
      }
      result += '\\';
    }
    result += unquoted[ii];
  }

  return result;
}

void Regex::Impl::printError(int error) {
  char buffer[512];
  pcre2_get_error_message_8(error, reinterpret_cast<PCRE2_UCHAR8 *>(buffer), sizeof(buffer));
  LOGE("pcre2 error: %s", buffer);
}

Regex::Regex(std::string_view pattern) { impl_ = std::make_unique<Impl>(pattern); }

Regex::~Regex() = default;

bool Regex::valid() const { return impl_->valid(); }

void Regex::matchAll(std::vector<Range> &ret, std::string_view str) const { impl_->matchAll(ret, str); }

std::string Regex::quoteMeta(std::string_view unquoted) { return Impl::quoteMeta(unquoted); }

}  // namespace tinygpt::tokenizer
