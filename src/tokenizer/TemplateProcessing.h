/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Base.h"

namespace tinygpt::tokenizer {

struct TemplateElement {
  enum Type { SpecialToken, Sequence } type;
  std::string id;
  int typeId;
};

class TemplateProcessing : public Component {
 public:
  TemplateProcessing(const std::vector<TemplateElement>& single, const std::vector<TemplateElement>& pair,
                     const ankerl::unordered_dense::map<std::string, std::vector<int32_t>>& specialTokens);

  ComponentType getType() override { return ComponentType::TEMPLATE_PROCESSING; }
  std::vector<int32_t> postProcess(const std::vector<int32_t>& ids, bool addSpecialTokens) override;

 private:
  void processTemplate(std::vector<int32_t>& outIds, std::vector<int32_t>& outTypeIds,
                       const std::vector<TemplateElement>& tmpl, const std::vector<int32_t>& idsA,
                       const std::vector<int32_t>& idsB, bool addSpecialTokens) const;

  std::vector<TemplateElement> single_;
  std::vector<TemplateElement> pair_;
  ankerl::unordered_dense::map<std::string, std::vector<int32_t>> specialTokens_;

  size_t extraCntSingle_;
  size_t extraCntPair_;
};

}  // namespace tinygpt::tokenizer
