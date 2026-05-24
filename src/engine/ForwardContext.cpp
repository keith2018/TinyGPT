/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "ForwardContext.h"

namespace tinygpt {

namespace {

thread_local ForwardContext* g_ctx = nullptr;

}  // namespace

ForwardContext* ForwardContext::current() { return g_ctx; }

ForwardContext* ForwardContext::setCurrent(ForwardContext* ctx) {
  auto* prev = g_ctx;
  g_ctx = ctx;
  return prev;
}

}  // namespace tinygpt
