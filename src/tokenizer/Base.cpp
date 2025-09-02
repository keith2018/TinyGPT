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

StringPieces ComponentSequence::preTokenize(const StringPieces& text) {
  StringPieces ret = text;
  for (auto& comp : components) {
    ret = comp->preTokenize(ret);
  }
  return ret;
}

std::vector<int32_t> ComponentSequence::postProcess(const std::vector<int32_t>& ids, bool addSpecialTokens) {
  std::vector<int32_t> currIds = ids;
  for (auto& comp : components) {
    currIds = comp->postProcess(currIds, addSpecialTokens);
  }
  return currIds;
}

std::vector<std::string> ComponentSequence::decode(const std::vector<std::string>& pieces) {
  std::vector<std::string> currPieces = pieces;
  for (auto& comp : components) {
    currPieces = comp->decode(currPieces);
  }
  return currPieces;
}

}  // namespace tinygpt::tokenizer
