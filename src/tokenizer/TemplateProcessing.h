/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Base.h"

namespace tinygpt::tokenizer {

class TemplateProcessing : public Component {
 public:
  TemplateProcessing(const std::vector<TemplateElement>& single, const std::vector<TemplateElement>& pair,
                     const ankerl::unordered_dense::map<std::string, std::vector<int32_t>>& specialTokens);

  ComponentType getType() override { return ComponentType::TEMPLATE_PROCESSING; }
  std::vector<int32_t> postProcess(const std::vector<int32_t>& ids) override;

 private:
  void processTemplate(std::vector<int32_t>& outIds, std::vector<int32_t>& outTypeIds,
                       const std::vector<TemplateElement>& tmpl, const std::vector<int32_t>& idsA,
                       const std::vector<int32_t>& idsB) const;

  std::vector<TemplateElement> single_;
  std::vector<TemplateElement> pair_;
  ankerl::unordered_dense::map<std::string, std::vector<int32_t>> specialTokens_;
};

}  // namespace tinygpt::tokenizer
