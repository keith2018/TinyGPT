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

PYBIND11_MODULE(_tinygpt, m) {
  // Tokenizer
  py::class_<tokenizer::Tokenizer>(m, "Tokenizer")
      .def(py::init<>())
      .def("initWithConfigHF", &tokenizer::Tokenizer::initWithConfigHF, py::arg("tokenizerPath"), py::arg("cfgPath"))
      .def("initWithConfigGPT2", &tokenizer::Tokenizer::initWithConfigGPT2, py::arg("encoderPath"),
           py::arg("vocabPath"))
      .def("token2Id", &tokenizer::Tokenizer::token2Id, py::arg("token"))
      .def("id2Token", &tokenizer::Tokenizer::id2Token, py::arg("id"))
      .def("encode", &tokenizer::Tokenizer::encode, py::arg("text"), py::arg("allowAddedTokens") = true)
      .def("encodeBatch", &tokenizer::Tokenizer::encodeBatch, py::arg("texts"), py::arg("numThreads") = 8,
           py::arg("allowAddedTokens") = true)
      .def("decode", &tokenizer::Tokenizer::decode, py::arg("ids"))
      .def("decodeBatch", &tokenizer::Tokenizer::decodeBatch, py::arg("ids"), py::arg("numThreads") = 8)
      .def_property_readonly("bosTokenId", &tokenizer::Tokenizer::bosTokenId)
      .def_property_readonly("eosTokenId", &tokenizer::Tokenizer::eosTokenId)
      .def_property_readonly("padTokenId", &tokenizer::Tokenizer::padTokenId)
      .def("bosTokenStr", &tokenizer::Tokenizer::bosTokenStr)
      .def("eosTokenStr", &tokenizer::Tokenizer::eosTokenStr)
      .def("padTokenStr", &tokenizer::Tokenizer::padTokenStr);
}
