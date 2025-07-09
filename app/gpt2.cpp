/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include <cassert>
#include <cstdio>

#include "Model.h"
#include "tokenizer/Tokenizer.h"

#define GPT2_HPARAMS_PATH "assets/gpt2/hparams.json"
#define GPT2_MODEL_DICT_PATH "assets/gpt2/model_index.json"
#define GPT2_ENCODER_PATH "assets/gpt2/encoder.json"
#define GPT2_VOCAB_PATH "assets/gpt2/vocab.bpe"

#define MAX_INPUT_LEN 64
#define MAX_GPT_TOKEN 64
#define TOKEN_END_LINE 198

#define INPUT_EXIT "exit\n"

using namespace tinygpt;

// int32_t -> int64_t
std::vector<int64_t> int32ToInt64(const std::vector<int32_t>& input) {
  std::vector<int64_t> output;
  output.reserve(input.size());
  for (int32_t v : input) {
    output.push_back(v);
  }
  return output;
}

void app_gpt2() {
  LOGI("app_gpt2()");

  GPT2 gpt2;

  // load model
  bool ret = Model::loadModelGPT2(gpt2, GPT2_HPARAMS_PATH, GPT2_MODEL_DICT_PATH);
  if (!ret) {
    LOGE("load GPT2 model failed !");
    return;
  }

  assert(MAX_GPT_TOKEN < gpt2.hparams.n_ctx);

  // init tokenizer
  tokenizer::Tokenizer tokenizer;
  bool initOk = tokenizer.initWithConfigGPT2(GPT2_ENCODER_PATH, GPT2_VOCAB_PATH);
  if (!initOk) {
    LOGE("init tokenizer failed !");
    return;
  }

  std::printf("INPUT:");
  char inputChars[MAX_INPUT_LEN];

  while (std::fgets(inputChars, MAX_INPUT_LEN, stdin)) {
    std::string inputStr(inputChars);

    // check exit
    if (inputStr == INPUT_EXIT) {
      break;
    }

    auto tokens = int32ToInt64(tokenizer.encode(inputStr));
    auto maxTokens = MAX_GPT_TOKEN - tokens.size();
    bool skipHeadBlank = true;

    // generate answers
    std::printf("GPT:");
    Model::generate(tokens, gpt2.params, gpt2.hparams.n_head, maxTokens, [&](int64_t token) -> bool {
      if (skipHeadBlank && token == TOKEN_END_LINE) {
        return false;
      }
      skipHeadBlank = false;
      if (token == TOKEN_END_LINE) {
        return true;
      }
      auto outputText = tokenizer.decode({static_cast<int32_t>(token)});
      std::printf("%s", outputText.c_str());
      std::fflush(stdout);
      return false;
    });
    std::printf("\nINPUT:");
  }
}