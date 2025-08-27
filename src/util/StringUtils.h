/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <string>

namespace tinygpt {

class StringUtils {
 public:
  static std::string repr(const std::string& s) {
    std::string result;
    result.reserve(s.size());

    for (char c : s) {
      switch (c) {
        case '\n':
          result += "\\n";
          break;
        case '\t':
          result += "\\t";
          break;
        case '\r':
          result += "\\r";
          break;
        case '\"':
          result += "\\\"";
          break;
        case '\\':
          result += "\\\\";
          break;
        default:
          result += c;
      }
    }
    return result;
  }
};

}  // namespace tinygpt