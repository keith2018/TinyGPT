/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "Base.h"

namespace tinygpt::tokenizer {

void ComponentSequence::addComponent(std::unique_ptr<Component>&& component) {
  if (component) {
    components.push_back(std::move(component));
  }
}

PreTokenizedString ComponentSequence::preTokenize(std::string_view text) {
  std::vector<std::string_view> buffer1 = {text};
  std::vector<std::string_view> buffer2;
  std::vector<std::string_view>* curr = &buffer1;
  std::vector<std::string_view>* next = &buffer2;

  std::vector<std::vector<std::string>> backStrings(components.size());

  for (size_t i = 0; i < components.size(); i++) {
    auto& comp = components[i];
    auto& backString = backStrings[i];
    next->clear();
    backString.reserve(curr->size());
    for (const auto& t : *curr) {
      auto compRet = comp->preTokenize(t);
      backString.push_back(std::move(compRet.backStr));
      auto& backStr = backString.back();
      next->reserve(compRet.pieces.size() * curr->size());
      for (const auto& r : compRet.pieces) {
        next->emplace_back(backStr.data() + r.first, r.second - r.first);
      }
    }
    std::swap(curr, next);
  }

  PreTokenizedString ret;
  size_t strLen = 0;
  for (const auto& token : *curr) {
    strLen += token.size();
  }
  ret.pieces.reserve(curr->size());
  ret.backStr.reserve(strLen);

  for (const auto& token : *curr) {
    const auto pos = ret.backStr.size();
    ret.backStr.append(token);
    ret.pieces.emplace_back(pos, pos + token.size());
  }
  return ret;
}

std::vector<int32_t> ComponentSequence::postProcess(const std::vector<int32_t>& ids) {
  std::vector<int32_t> currIds = ids;
  for (auto& comp : components) {
    currIds = comp->postProcess(currIds);
  }
  return currIds;
}

}  // namespace tinygpt::tokenizer
