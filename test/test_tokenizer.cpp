/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "test.h"
#include "Tokenizer.h"

using namespace TinyGPT;

static Encoder getEncoder() {
  static Encoder encoder = Encoder::getEncoder("assets/gpt2");
  return encoder;
}

static std::string str0 = "hello world!";
static std::vector<int32_t> tokens0 = {31373, 995, 0};

static std::string str1 = "Thanks for putting me into the right direction";
static std::vector<int32_t> tokens1 = {9690, 329, 5137, 502, 656, 262, 826, 4571};

static std::string str2 = "hello，你好啊, thanks";
static std::vector<int32_t> tokens2 = {31373, 171, 120, 234, 19526, 254, 25001, 121, 161, 243, 232, 11, 5176};

static std::string str3 = " ありがとうございます。 Arigatoo gozaimasu";
static std::vector<int32_t> tokens3 = {23294, 224, 28255, 35585, 30201, 29557, 2515, 242, 2515, 244, 18566, 30159,
                                       33623, 16764, 943, 328, 265, 2238, 467, 89, 1385, 27345};

TEST(TEST_TOKENIZER, encode) {
  auto encoder = getEncoder();
  EXPECT_TRUE(encoder.encode(str0) == tokens0);
  EXPECT_TRUE(encoder.encode(str1) == tokens1);
  EXPECT_TRUE(encoder.encode(str2) == tokens2);
  EXPECT_TRUE(encoder.encode(str3) == tokens3);
}

TEST(TEST_TOKENIZER, decode) {
  auto encoder = getEncoder();
  EXPECT_TRUE(encoder.decode(tokens0) == str0);
  EXPECT_TRUE(encoder.decode(tokens1) == str1);
  EXPECT_TRUE(encoder.decode(tokens2) == str2);
  EXPECT_TRUE(encoder.decode(tokens3) == str3);
}
