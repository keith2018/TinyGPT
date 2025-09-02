/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "Fuse.h"

namespace tinygpt::tokenizer {

std::vector<std::string> Fuse::decode(const std::vector<std::string>& pieces) {
  std::string ret;
  size_t len = 0;
  for (auto& s : pieces) {
    len += s.size();
  }
  ret.reserve(len);
  for (auto& s : pieces) {
    ret.append(s);
  }
  return {ret};
}

}  // namespace tinygpt::tokenizer