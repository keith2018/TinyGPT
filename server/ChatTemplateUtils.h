/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <string>

namespace tinygpt::server {

// requires <|im_start|>/<|im_end|> in the vocabulary
extern const char* kDefaultChatMLTemplate;

std::string stripChatMLTags(const std::string& text);

}  // namespace tinygpt::server
