/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "SafeTensors.h"
#include "Utils/Timer.h"
#include "models/GPT2.h"
#include "tokenizer/Tokenizer.h"

const std::string GPT2_MODEL_DIR = "assets/gpt2";  // clone from https://huggingface.co/openai-community/gpt2
const std::string INPUT_STR = "Alan Turing theorized that computers would one day become";
constexpr int64_t MAX_TOKENS = 32;

using namespace tinygpt;

// TODO
template <typename TypeSrc, typename TypeDst>
static std::vector<TypeDst> vecCvt(const std::vector<TypeSrc>& input) {
  std::vector<TypeDst> output;
  output.reserve(input.size());
  for (auto v : input) {
    output.push_back(static_cast<TypeDst>(v));
  }
  return output;
}

void demo_gpt2() {
  LOGI("demo_gpt2()");

  Timer timer;
  timer.start();

  gpt2::GPT2Config config;
  gpt2::GPT2LMHeadModel model(config);
  bool loadOk = SafeTensors::load(model.transformer, GPT2_MODEL_DIR + "/model.safetensors");
  if (!loadOk) {
    return;
  }

  auto device = Device::cuda();
  model.to(device);

  tokenizer::Tokenizer tokenizer;
  loadOk = tokenizer.initWithConfigHF(GPT2_MODEL_DIR + "/tokenizer.json", GPT2_MODEL_DIR + "/tokenizer_config.json");
  if (!loadOk) {
    return;
  }

  auto input_ids = tokenizer.encode(INPUT_STR);
  Options idxOptions = options::dtype(DType::Int64).device(device);
  auto input = Tensor(vecCvt<int32_t, int64_t>(input_ids), idxOptions);
  input.unsqueeze_(0);

  Sampler sampler({0.8f, 10, 0.8});

  auto ids = gpt2::generate(config, model, sampler, input, MAX_TOKENS);
  auto output = tokenizer.decode(vecCvt<int64_t, int32_t>(ids.toList<int64_t>()));
  LOGD("output: %s", output.c_str());

  timer.mark();
  LOGD("Total Time cost: %lld ms", timer.elapseMillis());
}
