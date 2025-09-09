/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "TokenizerConfig.h"

#include <fstream>

#include "JsonHelper.h"
#include "tokenizer/BPE.h"
#include "tokenizer/ByteFallback.h"
#include "tokenizer/ByteLevel.h"
#include "tokenizer/Fuse.h"
#include "tokenizer/Metaspace.h"
#include "tokenizer/Replace.h"
#include "tokenizer/Strip.h"
#include "tokenizer/UnicodeNorm.h"

namespace tinygpt::huggingface::tokenizer {

using namespace tinygpt::tokenizer;
using namespace tinygpt::json;

static ComponentType parseComponentType(const std::string& s) {
  if (s == "BPE") return ComponentType::BPE;
  if (s == "ByteFallback") return ComponentType::BYTE_FALLBACK;
  if (s == "ByteLevel") return ComponentType::BYTE_LEVEL;
  if (s == "Fuse") return ComponentType::FUSE;
  if (s == "Metaspace") return ComponentType::METASPACE;
  if (s == "NFC") return ComponentType::NFC;
  if (s == "NFD") return ComponentType::NFD;
  if (s == "NFKC") return ComponentType::NFKC;
  if (s == "NFKD") return ComponentType::NFKD;
  if (s == "Replace") return ComponentType::REPLACE;
  if (s == "Sequence") return ComponentType::SEQUENCE;
  if (s == "Split") return ComponentType::SPLIT;
  if (s == "Strip") return ComponentType::STRIP;
  if (s == "TemplateProcessing") return ComponentType::TEMPLATE_PROCESSING;
  return ComponentType::UNKNOWN;
}

static SplitDelimiterBehavior parseSplitDelimiterBehavior(const std::string& s) {
  if (s == "Removed") return SplitDelimiterBehavior::REMOVED;
  if (s == "Isolated") return SplitDelimiterBehavior::ISOLATED;
  if (s == "MergedWithPrevious") return SplitDelimiterBehavior::MERGED_WITH_PREVIOUS;
  if (s == "MergedWithNext") return SplitDelimiterBehavior::MERGED_WITH_NEXT;
  if (s == "Contiguous") return SplitDelimiterBehavior::CONTIGUOUS;
  return SplitDelimiterBehavior::UNKNOWN;
}

static ConfigAddedToken parseConfigAddedToken(const rapidjson::Value& j) {
  ConfigAddedToken t;
  t.id = getJsonValue(j, "id", -1);
  t.content = j.IsString() ? j.GetString() : getJsonValue(j, "content", std::string(""));
  t.singleWord = getJsonValue(j, "single_word", false);
  t.lStrip = getJsonValue(j, "lstrip", false);
  t.rStrip = getJsonValue(j, "rstrip", false);
  t.normalized = getJsonValue(j, "normalized", false);
  t.special = getJsonValue(j, "special", false);
  return t;
}

static std::unique_ptr<Config> parseConfigByteLevel(const rapidjson::Value& j) {
  auto c = std::make_unique<ConfigByteLevel>();
  c->type = ComponentType::BYTE_LEVEL;
  c->addPrefixSpace = getJsonValue(j, "add_prefix_space", false);
  c->trimOffsets = getJsonValue(j, "trim_offsets", false);
  c->useRegex = getJsonValue(j, "use_regex", false);
  return c;
}

static std::unique_ptr<Config> parseConfigSplit(const rapidjson::Value& j) {
  auto c = std::make_unique<ConfigSplit>();
  c->type = ComponentType::SPLIT;

  if (j.HasMember("pattern")) {
    const auto& pattern = j["pattern"];
    if (pattern.IsObject()) {
      for (auto it = pattern.MemberBegin(); it != pattern.MemberEnd(); /* ++it */) {
        if (it->value.IsString()) {
          c->pattern = std::string(it->value.GetString(), it->value.GetStringLength());
        }
        break;  // only pick the first one
      }
    } else if (pattern.IsString()) {
      c->pattern = std::string(pattern.GetString(), pattern.GetStringLength());
    }
  }

  std::string behavior = getJsonValue(j, "behavior", std::string("Removed"));
  c->behavior = parseSplitDelimiterBehavior(behavior);
  c->invert = getJsonValue(j, "invert", false);
  return c;
}

static std::unique_ptr<Config> parseConfigBPE(const rapidjson::Value& j) {
  auto c = std::make_unique<ConfigBPE>();
  c->type = ComponentType::BPE;
  c->ignoreMerges = getJsonValue(j, "ignore_merges", false);

  // vocab
  if (j.HasMember("vocab")) {
    auto& jsonVocab = j["vocab"];
    c->vocab.reserve(jsonVocab.MemberCount());
    for (auto it = jsonVocab.MemberBegin(); it != jsonVocab.MemberEnd(); ++it) {
      c->vocab[it->name.GetString()] = it->value.GetInt();
    }
  }

  // merges
  if (j.HasMember("merges")) {
    auto& jsonMerges = j["merges"];
    c->merges.reserve(jsonMerges.Size());
    int32_t idx = 0;
    for (const auto& m : jsonMerges.GetArray()) {
      if (m.IsString()) {
        std::string_view s = m.GetString();
        size_t pos = s.find(' ');
        if (pos != std::string::npos) {
          c->merges[{std::string(s.substr(0, pos)), std::string(s.substr(pos + 1))}] = idx;
          idx++;
        }
      } else if (m.IsArray()) {
        auto arr = m.GetArray();
        ASSERT(arr.Size() == 2);
        auto k = arr[0].GetString();
        auto v = arr[1].GetString();
        c->merges[{k, v}] = idx;
        idx++;
      }
    }
  }
  return c;
}

static TemplateElement parseTemplateElement(const rapidjson::Value& j) {
  if (j.HasMember("SpecialToken")) {
    const auto& st = j["SpecialToken"];
    TemplateElement e;
    e.type = TemplateElement::SpecialToken;
    e.id = st["id"].GetString();
    e.typeId = st["type_id"].GetInt();
    return e;
  }

  if (j.HasMember("Sequence")) {
    const auto& seq = j["Sequence"];
    TemplateElement e;
    e.type = TemplateElement::Sequence;
    e.id = seq["id"].GetString();  // "A" or "B"
    e.typeId = seq["type_id"].GetInt();
    return e;
  }

  LOGE("Unknown template element");
  return {};
}

static std::unique_ptr<Config> parseConfigTemplateProcessing(const rapidjson::Value& j) {
  auto c = std::make_unique<ConfigTemplateProcessing>();
  c->type = ComponentType::TEMPLATE_PROCESSING;
  const auto& st = j["special_tokens"];
  for (auto it = st.MemberBegin(); it != st.MemberEnd(); ++it) {
    std::string name = it->name.GetString();
    const auto& ids_arr = it->value["ids"];
    std::vector<int32_t> ids;
    for (auto& v : ids_arr.GetArray()) {
      ids.push_back(v.GetInt());
    }
    c->specialTokens[name] = {ids};
  }

  // single
  for (auto& elem : j["single"].GetArray()) {
    c->single.push_back(parseTemplateElement(elem));
  }
  // pair
  for (auto& elem : j["pair"].GetArray()) {
    c->pair.push_back(parseTemplateElement(elem));
  }

  return c;
}

static std::unique_ptr<Config> parseConfigMetaspace(const rapidjson::Value& j) {
  auto c = std::make_unique<ConfigMetaspace>();
  c->type = ComponentType::METASPACE;
  c->replacement = getJsonValue(j, "replacement", std::string("\u2581"));
  c->prependScheme = getJsonValue(j, "prepend_scheme", std::string("always"));
  c->split = getJsonValue(j, "split", true);
  return c;
}

static std::unique_ptr<Config> parseConfigReplace(const rapidjson::Value& j) {
  auto c = std::make_unique<ConfigReplace>();
  c->type = ComponentType::REPLACE;
  const auto& pt = j["pattern"];
  if (pt.HasMember("String")) {
    c->patternString = getJsonValue(pt, "String", std::string(""));
  }
  if (pt.HasMember("Regex")) {
    c->patternRegex = getJsonValue(pt, "Regex", std::string(""));
  }
  c->content = getJsonValue(j, "content", std::string(""));
  return c;
}

static std::unique_ptr<Config> parseConfigStrip(const rapidjson::Value& j) {
  auto c = std::make_unique<ConfigStrip>();
  c->type = ComponentType::STRIP;
  c->content = getJsonValue(j, "content", std::string(""));
  c->start = getJsonValue(j, "start", 0);
  c->stop = getJsonValue(j, "stop", 0);
  return c;
}

static std::unique_ptr<Config> parseConfig(const rapidjson::Value& j, const std::string& defaultType = {});

static std::unique_ptr<Config> parseConfigSequence(const rapidjson::Value& j) {  // NOLINT(misc-no-recursion)
  auto c = std::make_unique<ConfigSequence>();
  c->type = ComponentType::SEQUENCE;
  for (auto it = j.MemberBegin(); it != j.MemberEnd(); ++it) {
    std::string key = it->name.GetString();
    if (key == "type") continue;
    if (it->value.IsArray()) {
      for (const auto& sub : it->value.GetArray()) {
        c->configs.push_back(parseConfig(sub));
      }
    }
  }
  return c;
}

static std::unique_ptr<Config> parseConfigNoParams(const ComponentType& type) {
  auto c = std::make_unique<Config>();
  c->type = type;
  return c;
}

// NOLINTNEXTLINE(misc-no-recursion)
static std::unique_ptr<Config> parseConfig(const rapidjson::Value& j, const std::string& defaultType) {
  if (j.IsNull()) return nullptr;
  std::string typeStr = getJsonValue(j, "type", std::string(""));
  if (typeStr.empty()) {
    typeStr = defaultType;
  }
  ComponentType type = parseComponentType(typeStr);
  switch (type) {
    case ComponentType::BPE:
      return parseConfigBPE(j);
    case ComponentType::BYTE_FALLBACK:
    case ComponentType::FUSE:
    case ComponentType::NFC:
    case ComponentType::NFD:
    case ComponentType::NFKC:
    case ComponentType::NFKD:
      return parseConfigNoParams(type);
    case ComponentType::BYTE_LEVEL:
      return parseConfigByteLevel(j);
    case ComponentType::METASPACE:
      return parseConfigMetaspace(j);
    case ComponentType::REPLACE:
      return parseConfigReplace(j);
    case ComponentType::SEQUENCE:
      return parseConfigSequence(j);
    case ComponentType::SPLIT:
      return parseConfigSplit(j);
    case ComponentType::STRIP:
      return parseConfigStrip(j);
    case ComponentType::TEMPLATE_PROCESSING:
      return parseConfigTemplateProcessing(j);
    default:
      LOGE("Component type not support: %s", typeStr.c_str());
  }
  return nullptr;
}

static bool loadTokenizer(TokenizerConfig& cfg, const std::string& tokenizerPath) {
  std::ifstream in(tokenizerPath, std::ios::binary);
  if (!in) {
    LOGE("Cannot open file: %s", tokenizerPath.c_str());
    return false;
  }
  std::string jsonStr((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  rapidjson::Document j;
  if (j.Parse(jsonStr.c_str()).HasParseError()) {
    LOGE("Parse tokenizer error");
    return false;
  }

  if (!j.IsObject() || j.ObjectEmpty()) {
    LOGE("Tokenizer file empty: %s", tokenizerPath.c_str());
    return false;
  }

  // version
  cfg.version = getJsonValue(j, "version", std::string(""));

  // added_tokens
  if (j.HasMember("added_tokens") && j["added_tokens"].IsArray()) {
    for (const auto& tok : j["added_tokens"].GetArray()) {
      cfg.addedTokens.push_back(parseConfigAddedToken(tok));
    }
  }

  // normalizer
  if (j.HasMember("normalizer")) {
    cfg.normalizer = parseConfig(j["normalizer"]);
  }

  // pre_tokenizer
  if (j.HasMember("pre_tokenizer")) {
    cfg.preTokenizer = parseConfig(j["pre_tokenizer"]);
  }

  // model
  if (j.HasMember("model")) {
    cfg.model = parseConfig(j["model"], "BPE");
  }

  // post_processor
  if (j.HasMember("post_processor")) {
    cfg.postProcessor = parseConfig(j["post_processor"]);
  }

  // decoder
  if (j.HasMember("decoder")) {
    cfg.decoder = parseConfig(j["decoder"]);
  }

  return true;
}

static bool loadConfig(TokenizerConfig& cfg, const std::string& cfgPath) {
  std::ifstream in(cfgPath, std::ios::binary);
  if (!in) {
    LOGE("Cannot open file: %s", cfgPath.c_str());
    return false;
  }
  std::string jsonStr((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  rapidjson::Document j;
  if (j.Parse(jsonStr.c_str()).HasParseError()) {
    LOGE("Parse config error");
    return false;
  }

  if (!j.IsObject() || j.ObjectEmpty()) {
    LOGE("Config file empty: %s", cfgPath.c_str());
    return false;
  }

  cfg.addBosToken = getJsonValue(j, "add_bos_token", false);
  cfg.addEosToken = getJsonValue(j, "add_eos_token", false);

  if (j.HasMember("bos_token")) cfg.bosToken = parseConfigAddedToken(j["bos_token"]);
  if (j.HasMember("eos_token")) cfg.eosToken = parseConfigAddedToken(j["eos_token"]);
  if (j.HasMember("pad_token")) cfg.padToken = parseConfigAddedToken(j["pad_token"]);

  cfg.modelMaxLength = getJsonValue(j, "model_max_length", 0);
  cfg.chatTemplate = getJsonValue(j, "chat_template", std::string(""));

  return true;
}

bool load(TokenizerConfig& cfg, const std::string& tokenizerPath, const std::string& cfgPath) {
  return loadTokenizer(cfg, tokenizerPath) && loadConfig(cfg, cfgPath);
}

static std::unique_ptr<Component> createSequence(const std::unique_ptr<Config>& cfg) {  // NOLINT(misc-no-recursion)
  if (!cfg) {
    return nullptr;
  }
  auto* config = dynamic_cast<ConfigSequence*>(cfg.get());
  auto seq = std::make_unique<ComponentSequence>();
  for (auto& subCfg : config->configs) {
    seq->addComponent(createComponent(subCfg));
  }
  return std::move(seq);
}

static std::unique_ptr<Component> createSplit(const std::unique_ptr<Config>& cfg) {
  if (!cfg) {
    return nullptr;
  }
  auto* config = dynamic_cast<ConfigSplit*>(cfg.get());
  auto split = std::make_unique<Split>(config->pattern, config->behavior, config->invert);
  return std::move(split);
}

static std::unique_ptr<Component> createByteLevel(const std::unique_ptr<Config>& cfg) {
  if (!cfg) {
    return nullptr;
  }
  auto* config = dynamic_cast<ConfigByteLevel*>(cfg.get());
  auto byteLevel = std::make_unique<ByteLevel>(config->addPrefixSpace, config->useRegex);  // ignore 'trimOffsets'
  return std::move(byteLevel);
}

static std::unique_ptr<Component> createBPE(const std::unique_ptr<Config>& cfg) {
  if (!cfg) {
    return nullptr;
  }
  auto* config = dynamic_cast<ConfigBPE*>(cfg.get());
  auto bpe = std::make_unique<BPE>(config->vocab, config->merges, config->ignoreMerges);
  return std::move(bpe);
}

static std::unique_ptr<Component> createTemplateProcessing(const std::unique_ptr<Config>& cfg) {
  if (!cfg) {
    return nullptr;
  }
  auto* config = dynamic_cast<ConfigTemplateProcessing*>(cfg.get());
  auto tp = std::make_unique<TemplateProcessing>(config->single, config->pair, config->specialTokens);
  return std::move(tp);
}

static std::unique_ptr<Component> createMetaspace(const std::unique_ptr<Config>& cfg) {
  if (!cfg) {
    return nullptr;
  }
  auto* config = dynamic_cast<ConfigMetaspace*>(cfg.get());
  auto tp = std::make_unique<Metaspace>(config->replacement, config->prependScheme, config->split);
  return std::move(tp);
}

static std::unique_ptr<Component> createReplace(const std::unique_ptr<Config>& cfg) {
  if (!cfg) {
    return nullptr;
  }
  auto* config = dynamic_cast<ConfigReplace*>(cfg.get());
  auto tp = std::make_unique<Replace>(config->patternString, config->patternRegex, config->content);
  return std::move(tp);
}

static std::unique_ptr<Component> createStrip(const std::unique_ptr<Config>& cfg) {
  if (!cfg) {
    return nullptr;
  }
  auto* config = dynamic_cast<ConfigStrip*>(cfg.get());
  auto tp = std::make_unique<Strip>(config->content, config->start, config->stop);
  return std::move(tp);
}

static std::unique_ptr<Component> createNoParams(const std::unique_ptr<Config>& cfg) {
  if (!cfg) {
    return nullptr;
  }
  switch (cfg->type) {
    case ComponentType::BYTE_FALLBACK:
      return std::make_unique<ByteFallback>();
    case ComponentType::FUSE:
      return std::make_unique<Fuse>();
    case ComponentType::NFC:
      return std::make_unique<NFC>();
    case ComponentType::NFD:
      return std::make_unique<NFD>();
    case ComponentType::NFKC:
      return std::make_unique<NFKC>();
    case ComponentType::NFKD:
      return std::make_unique<NFKD>();
    default:
      break;
  }
  return nullptr;
}

std::unique_ptr<Component> createComponent(const std::unique_ptr<Config>& cfg) {  // NOLINT(misc-no-recursion)
  if (!cfg) {
    return nullptr;
  }
  switch (cfg->type) {
    case ComponentType::BPE:
      return createBPE(cfg);
    case ComponentType::BYTE_FALLBACK:
    case ComponentType::FUSE:
    case ComponentType::NFC:
    case ComponentType::NFD:
    case ComponentType::NFKC:
    case ComponentType::NFKD:
      return createNoParams(cfg);
    case ComponentType::BYTE_LEVEL:
      return createByteLevel(cfg);
    case ComponentType::METASPACE:
      return createMetaspace(cfg);
    case ComponentType::REPLACE:
      return createReplace(cfg);
    case ComponentType::SEQUENCE:
      return createSequence(cfg);
    case ComponentType::SPLIT:
      return createSplit(cfg);
    case ComponentType::STRIP:
      return createStrip(cfg);
    case ComponentType::TEMPLATE_PROCESSING:
      return createTemplateProcessing(cfg);
    default:
      LOGE("Unknown config type: %d", cfg->type);
  }
  return nullptr;
}

}  // namespace tinygpt::huggingface::tokenizer
