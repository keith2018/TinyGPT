/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "ChatTemplateUtils.h"

namespace tinygpt::server {

const char* kDefaultChatMLTemplate =
    "{%- for message in messages -%}"
    "{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{- '<|im_start|>assistant\n' -}}"
    "{%- endif -%}";

std::string stripChatMLTags(const std::string& text) {
  std::string result = text;
  // strip trailing <|im_end|>
  const std::string imEnd = "<|im_end|>";
  while (result.size() >= imEnd.size() && result.compare(result.size() - imEnd.size(), imEnd.size(), imEnd) == 0) {
    result.erase(result.size() - imEnd.size());
  }
  // strip trailing whitespace after removing tags
  while (!result.empty() && (result.back() == '\n' || result.back() == '\r' || result.back() == ' ')) {
    result.pop_back();
  }
  // if any <|im_start|> remains (model continued the conversation), truncate at it
  auto pos = result.find("<|im_start|>");
  if (pos != std::string::npos) {
    result = result.substr(0, pos);
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r' || result.back() == ' ')) {
      result.pop_back();
    }
  }
  return result;
}

}  // namespace tinygpt::server
