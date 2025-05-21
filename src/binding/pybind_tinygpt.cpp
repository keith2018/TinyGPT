/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tokenizer/Tokenizer.h"

namespace py = pybind11;
using namespace tinygpt;

// clang-format off

PYBIND11_MODULE(_tinygpt, m) {
  // Tokenizer
  py::class_<tokenizer::Tokenizer>(m, "Tokenizer")
      .def(py::init<>())
      .def("init_with_config_hf", &tokenizer::Tokenizer::initWithConfigHF, py::arg("tokenizer_path"), py::arg("cfg_path"))
      .def("init_with_config_gpt2", &tokenizer::Tokenizer::initWithConfigGPT2, py::arg("encoder_path"), py::arg("vocab_path"))
      .def("token_to_id", &tokenizer::Tokenizer::token2Id, py::arg("token"))
      .def("id_to_token", &tokenizer::Tokenizer::id2Token, py::arg("id"))
      .def("encode", &tokenizer::Tokenizer::encode, py::arg("text"), py::arg("allow_added_tokens") = true)
      .def("encode_batch", &tokenizer::Tokenizer::encodeBatch, py::arg("texts"), py::arg("num_threads") = 8, py::arg("allow_added_tokens") = true)
      .def("decode", &tokenizer::Tokenizer::decode, py::arg("ids"))
      .def("decode_batch", &tokenizer::Tokenizer::decodeBatch, py::arg("ids"), py::arg("num_threads") = 8)
      .def("decode_stream", &tokenizer::Tokenizer::decodeStream, py::arg("ids"))
      .def_property_readonly("bos_token_id", &tokenizer::Tokenizer::bosTokenId)
      .def_property_readonly("eos_token_id", &tokenizer::Tokenizer::eosTokenId)
      .def_property_readonly("pad_token_id", &tokenizer::Tokenizer::padTokenId)
      .def("bos_token_str", &tokenizer::Tokenizer::bosTokenStr)
      .def("eos_token_str", &tokenizer::Tokenizer::eosTokenStr)
      .def("pad_token_str", &tokenizer::Tokenizer::padTokenStr);
}

// clang-format on