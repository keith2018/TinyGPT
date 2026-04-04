/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "RequestTypes.h"
#include "rapidjson/document.h"

namespace tinygpt::server {

size_t incompleteUtf8Tail(const std::string& s);

// returns: {truncated_text, matched}.
std::pair<std::string, bool> checkStopStrings(const std::string& text, const std::vector<std::string>& stopStrings,
                                              bool includeStop);

std::string validateSamplingParams(const InferenceRequest& req);

void parseCommonInferenceParams(const rapidjson::Document& reqDoc, InferenceRequest& inferReq);

}  // namespace tinygpt::server
