/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include <vector>

#include "TinyTorch.h"
#include "Tensor/CachedAllocator.h"
#include "Utils/CUDAGraph.h"
#include "Utils/CUDAUtils.h"
#include "test.h"

namespace tt = tinytorch;

namespace tinygpt {

namespace {

#define SKIP_IF_NO_CUDA()                                         \
  do {                                                            \
    if (!tt::cuda::deviceAvailable()) {                           \
      GTEST_SKIP() << "CUDA device not available; skipping test"; \
    }                                                             \
  } while (0)

tt::Tensor makeCudaTensor(const std::vector<float>& data, const std::vector<int64_t>& shape) {
  auto cpuOpts = tt::Options(tt::Device::cpu(), tt::DType::Float32).noGrad();
  tt::Tensor host(data, shape, cpuOpts);
  return host.to(tt::Device(tt::DeviceType::CUDA, 0));
}

std::vector<float> toHost(const tt::Tensor& t) {
  auto cpu = t.to(tt::Device::cpu());
  return cpu.toList<float>();
}

}  // namespace

// =============================================================================
// CachedAllocator pool isolation tests
// =============================================================================

TEST(cuda_graph, allocator_pool_new_id) {
  // newPoolId() returns unique, monotonically increasing IDs
  int id1 = tt::CachedAllocator::newPoolId();
  int id2 = tt::CachedAllocator::newPoolId();
  int id3 = tt::CachedAllocator::newPoolId();
  EXPECT_LT(id1, id2);
  EXPECT_LT(id2, id3);
}

TEST(cuda_graph, allocator_pool_isolation) {
  SKIP_IF_NO_CUDA();

  auto* allocator = tt::getCUDACachedAllocator(0);
  ASSERT_NE(allocator, nullptr);

  // Verify initial state: no active pool
  EXPECT_EQ(allocator->activePoolId(), -1);

  int poolId = tt::CachedAllocator::newPoolId();

  // Begin pool allocation
  allocator->beginAllocateToPool(poolId);
  EXPECT_EQ(allocator->activePoolId(), poolId);

  // Allocate a tensor (should go to pool)
  auto opts = tt::Options(tt::Device(tt::DeviceType::CUDA, 0), tt::DType::Float32).noGrad();
  tt::Tensor poolTensor({1024}, opts);
  void* poolPtr = poolTensor.dataPtr<>();

  // End pool allocation
  allocator->endAllocateToPool();
  EXPECT_EQ(allocator->activePoolId(), -1);

  // Allocate in default pool (should NOT overlap with pool tensor)
  tt::Tensor defaultTensor({1024}, opts);
  void* defaultPtr = defaultTensor.dataPtr<>();
  EXPECT_NE(poolPtr, defaultPtr);
}

TEST(cuda_graph, allocator_deterministic_addresses) {
  SKIP_IF_NO_CUDA();

  auto* allocator = tt::getCUDACachedAllocator(0);
  ASSERT_NE(allocator, nullptr);

  int poolId = tt::CachedAllocator::newPoolId();
  auto opts = tt::Options(tt::Device(tt::DeviceType::CUDA, 0), tt::DType::Float32).noGrad();

  // First allocation sequence in pool
  allocator->beginAllocateToPool(poolId);
  tt::Tensor t1({256}, opts);
  void* addr1 = t1.dataPtr<>();
  tt::Tensor t2({512}, opts);
  void* addr2 = t2.dataPtr<>();
  allocator->endAllocateToPool();

  // Free tensors — memory returns to pool's free list
  t1 = tt::Tensor();
  t2 = tt::Tensor();

  // Second allocation sequence in same pool — should get same addresses
  allocator->beginAllocateToPool(poolId);
  tt::Tensor t3({256}, opts);
  void* addr3 = t3.dataPtr<>();
  tt::Tensor t4({512}, opts);
  void* addr4 = t4.dataPtr<>();
  allocator->endAllocateToPool();

  EXPECT_EQ(addr1, addr3);
  EXPECT_EQ(addr2, addr4);

  // Cleanup
  t3 = tt::Tensor();
  t4 = tt::Tensor();
  allocator->freePool(poolId);
}

TEST(cuda_graph, allocator_pool_cleanup) {
  SKIP_IF_NO_CUDA();

  auto* allocator = tt::getCUDACachedAllocator(0);
  ASSERT_NE(allocator, nullptr);

  int poolId = tt::CachedAllocator::newPoolId();
  auto opts = tt::Options(tt::Device(tt::DeviceType::CUDA, 0), tt::DType::Float32).noGrad();

  // Allocate in pool
  allocator->beginAllocateToPool(poolId);
  tt::Tensor t({4096}, opts);
  allocator->endAllocateToPool();

  // Free tensor (memory goes back to pool's free list, not base allocator)
  t = tt::Tensor();

  // freePool should release memory back to base allocator
  allocator->freePool(poolId);

  // Should not crash — pool is now gone
  allocator->freePool(poolId);  // no-op on already freed pool
}

// =============================================================================
// CUDAGraph capture/replay tests
// =============================================================================

TEST(cuda_graph, basic_capture_replay) {
  SKIP_IF_NO_CUDA();

  // Create static input/output tensors
  auto input = makeCudaTensor({1.0f, 2.0f, 3.0f, 4.0f}, {1, 4});
  auto weight = makeCudaTensor({2.0f, 2.0f, 2.0f, 2.0f}, {1, 4});

  auto& stream = tt::cuda::getCurrentCUDAStream(0);
  tt::cuda::CUDAGraph graph;
  tt::Tensor output;

  // Warmup phase 1: stabilize cuBLAS workspace (default pool, no capture)
  output = tt::function::matmul(input, weight.t());
  stream.synchronize();

  // Warmup phase 2: pre-warm graph pool (pool routing, no capture)
  // This populates the pool with cached blocks so capture won't need cudaMalloc.
  int poolId = tt::CachedAllocator::newPoolId();
  auto* allocator = tt::getCUDACachedAllocator(0);
  allocator->beginAllocateToPool(poolId);
  output = tt::function::matmul(input, weight.t());
  stream.synchronize();
  output = tt::Tensor();  // free back to pool's free list
  allocator->endAllocateToPool();

  // Capture (pool already has cached blocks, no cudaMalloc during capture)
  {
    tt::cuda::CUDAGraphCaptureGuard guard(graph, stream, poolId);
    output = tt::function::matmul(input, weight.t());
  }

  ASSERT_TRUE(graph.valid());

  // Replay
  graph.replay(stream);
  stream.synchronize();

  auto result = toHost(output);
  // dot([1,2,3,4], [2,2,2,2]) = 2+4+6+8 = 20
  EXPECT_NEAR(result[0], 20.0f, 1e-3f);

  // Update input and replay again
  auto newInput = makeCudaTensor({1.0f, 1.0f, 1.0f, 1.0f}, {1, 4});
  input.copy_(newInput);
  graph.replay(stream);
  stream.synchronize();

  result = toHost(output);
  // dot([1,1,1,1], [2,2,2,2]) = 8
  EXPECT_NEAR(result[0], 8.0f, 1e-3f);

  // Cleanup: must drop output tensor BEFORE freePool, because output's memory
  // lives in the graph pool. If freePool runs first, the pool's BlockPool is
  // destroyed; then output's destructor calls deallocate → dangling pool pointer.
  graph.reset();
  output = tt::Tensor();  // release pool memory back to pool's free list
  allocator->freePool(poolId);
}

TEST(cuda_graph, multiple_replays_consistency) {
  SKIP_IF_NO_CUDA();

  auto input = makeCudaTensor({3.0f, 4.0f}, {1, 2});
  auto weight = makeCudaTensor({1.0f, 1.0f}, {1, 2});

  auto& stream = tt::cuda::getCurrentCUDAStream(0);
  tt::cuda::CUDAGraph graph;
  tt::Tensor output;

  // Warmup phase 1: stabilize cuBLAS workspace
  output = tt::function::matmul(input, weight.t());
  stream.synchronize();

  // Warmup phase 2: pre-warm graph pool
  int poolId = tt::CachedAllocator::newPoolId();
  auto* allocator = tt::getCUDACachedAllocator(0);
  allocator->beginAllocateToPool(poolId);
  output = tt::function::matmul(input, weight.t());
  stream.synchronize();
  output = tt::Tensor();
  allocator->endAllocateToPool();

  // Capture
  {
    tt::cuda::CUDAGraphCaptureGuard guard(graph, stream, poolId);
    output = tt::function::matmul(input, weight.t());
  }

  ASSERT_TRUE(graph.valid());

  // Replay 50 times — output should be consistent
  for (int i = 0; i < 50; i++) {
    graph.replay(stream);
  }
  stream.synchronize();

  auto result = toHost(output);
  // dot([3,4], [1,1]) = 7
  EXPECT_NEAR(result[0], 7.0f, 1e-3f);

  graph.reset();
  output = tt::Tensor();
  tt::getCUDACachedAllocator(0)->freePool(poolId);
}

TEST(cuda_graph, capture_guard_raii) {
  SKIP_IF_NO_CUDA();

  auto* allocator = tt::getCUDACachedAllocator(0);
  ASSERT_NE(allocator, nullptr);

  auto& stream = tt::cuda::getCurrentCUDAStream(0);
  tt::cuda::CUDAGraph graph;
  int poolId = tt::CachedAllocator::newPoolId();

  auto input = makeCudaTensor({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2});
  tt::Tensor output;

  // Warmup phase 1: stabilize cuBLAS workspace
  output = tt::function::matmul(input, input.t());
  stream.synchronize();

  // Warmup phase 2: pre-warm graph pool
  allocator->beginAllocateToPool(poolId);
  output = tt::function::matmul(input, input.t());
  stream.synchronize();
  output = tt::Tensor();
  allocator->endAllocateToPool();

  // Verify guard properly manages allocator state
  EXPECT_EQ(allocator->activePoolId(), -1);
  {
    tt::cuda::CUDAGraphCaptureGuard guard(graph, stream, poolId);
    EXPECT_EQ(allocator->activePoolId(), poolId);
    output = tt::function::matmul(input, input.t());
  }
  // After guard scope: allocator back to default, graph is valid
  EXPECT_EQ(allocator->activePoolId(), -1);
  ASSERT_TRUE(graph.valid());

  // Replay to verify capture worked
  graph.replay(stream);
  stream.synchronize();

  auto result = toHost(output);
  // [[5,6],[7,8]] @ [[5,7],[6,8]] = [[61, 83],[83, 113]]
  EXPECT_NEAR(result[0], 61.0f, 1e-2f);
  EXPECT_NEAR(result[1], 83.0f, 1e-2f);
  EXPECT_NEAR(result[2], 83.0f, 1e-2f);
  EXPECT_NEAR(result[3], 113.0f, 1e-2f);

  graph.reset();
  output = tt::Tensor();
  allocator->freePool(poolId);
}

TEST(cuda_graph, graph_reset_and_recapture) {
  SKIP_IF_NO_CUDA();

  auto input = makeCudaTensor({2.0f, 3.0f}, {1, 2});
  auto weight = makeCudaTensor({4.0f, 5.0f}, {1, 2});

  auto& stream = tt::cuda::getCurrentCUDAStream(0);
  tt::cuda::CUDAGraph graph;
  tt::Tensor output;

  // Warmup phase 1: stabilize cuBLAS workspace
  output = tt::function::matmul(input, weight.t());
  stream.synchronize();

  // Pre-warm graph pool for first capture
  int poolId1 = tt::CachedAllocator::newPoolId();
  auto* allocator = tt::getCUDACachedAllocator(0);
  allocator->beginAllocateToPool(poolId1);
  output = tt::function::matmul(input, weight.t());
  stream.synchronize();
  output = tt::Tensor();
  allocator->endAllocateToPool();

  // First capture
  {
    tt::cuda::CUDAGraphCaptureGuard guard(graph, stream, poolId1);
    output = tt::function::matmul(input, weight.t());
  }
  ASSERT_TRUE(graph.valid());

  graph.replay(stream);
  stream.synchronize();
  auto result1 = toHost(output);
  EXPECT_NEAR(result1[0], 23.0f, 1e-3f);  // 2*4 + 3*5 = 23

  // Reset and recapture with different operation
  graph.reset();
  output = tt::Tensor();  // release first capture's output before freeing pool
  EXPECT_FALSE(graph.valid());
  tt::getCUDACachedAllocator(0)->freePool(poolId1);

  // Second capture: use addition instead
  auto bias = makeCudaTensor({10.0f, 20.0f}, {1, 2});
  int poolId2 = tt::CachedAllocator::newPoolId();

  // Pre-warm graph pool for second capture
  allocator->beginAllocateToPool(poolId2);
  auto output2 = input + bias;
  stream.synchronize();
  output2 = tt::Tensor();
  allocator->endAllocateToPool();

  {
    tt::cuda::CUDAGraphCaptureGuard guard(graph, stream, poolId2);
    output2 = input + bias;
  }
  ASSERT_TRUE(graph.valid());

  graph.replay(stream);
  stream.synchronize();
  auto result2 = toHost(output2);
  EXPECT_NEAR(result2[0], 12.0f, 1e-3f);  // 2+10
  EXPECT_NEAR(result2[1], 23.0f, 1e-3f);  // 3+20

  graph.reset();
  output2 = tt::Tensor();
  tt::getCUDACachedAllocator(0)->freePool(poolId2);
}

}  // namespace tinygpt
