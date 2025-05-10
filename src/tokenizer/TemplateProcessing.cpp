/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "TemplateProcessing.h"

namespace tinygpt::tokenizer {

TemplateProcessing::TemplateProcessing(
    const std::vector<TemplateElement>& single, const std::vector<TemplateElement>& pair,
    const ankerl::unordered_dense::map<std::string, std::vector<int32_t>>& specialTokens)
    : single_(single), pair_(pair), specialTokens_(specialTokens) {}

std::vector<int32_t> TemplateProcessing::postProcess(const std::vector<int32_t>& ids) {
  // TODO pair sentence
  std::vector<int32_t> outIds;
  std::vector<int32_t> outTypeIds;
  processTemplate(outIds, outTypeIds, single_, ids, {});
  return outIds;
}

void TemplateProcessing::processTemplate(std::vector<int32_t>& outIds, std::vector<int32_t>& outTypeIds,
                                         const std::vector<TemplateElement>& tmpl, const std::vector<int32_t>& idsA,
                                         const std::vector<int32_t>& idsB) const {
  constexpr char const* ID_A = "A";
  constexpr char const* ID_B = "B";

  for (const auto& elem : tmpl) {
    if (elem.type == TemplateElement::SpecialToken) {
      auto it = specialTokens_.find(elem.id);
      if (it == specialTokens_.end()) {
        LOGE("Unknown special token: %s", elem.id.c_str());
        break;
      }
      for (auto id : it->second) {
        outIds.push_back(id);
        outTypeIds.push_back(elem.typeId);
      }
    } else if (elem.type == TemplateElement::Sequence) {
      const std::vector<int32_t>* ids = nullptr;
      if (elem.id == ID_A) {
        ids = &idsA;
      } else if (elem.id == ID_B) {
        ids = &idsB;
      } else {
        LOGE("Unknown sequence id: %s", elem.id.c_str());
        break;
      }
      for (auto id : *ids) {
        outIds.push_back(id);
        outTypeIds.push_back(elem.typeId);
      }
    }
  }
}

}  // namespace tinygpt::tokenizer
