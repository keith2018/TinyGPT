/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "test.h"

namespace tinygpt {

void checkFloatVectorNear(const std::vector<float>& v1, const std::vector<float>& v2) {
  EXPECT_EQ(v1.size(), v2.size());
  for (size_t i = 0; i < v1.size(); i++) {
    EXPECT_FLOAT_NEAR(v1[i], v2[i]);
  }
}

std::vector<std::string> getStrings(const tokenizer::PreTokenizedString& input) {
  std::vector<std::string> result;
  result.reserve(input.pieces.size());
  for (const auto& r : input.pieces) {
    result.emplace_back(input.backStr.substr(r.first, r.second - r.first));
  }
  return result;
}

}  // namespace tinygpt
