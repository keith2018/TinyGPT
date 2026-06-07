/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <memory>

#include "Distributed/DistributedProcessGroup.h"

namespace tinygpt::distributed {

class Communicator {
 public:
  static Communicator& tp() {
    static Communicator c;
    return c;
  }

  void init(std::shared_ptr<tinytorch::distributed::ProcessGroup> pg) {
    pg_ = std::move(pg);
    rank_ = pg_->getRank();
    worldSize_ = pg_->getSize();
  }

  void reset() {
    pg_.reset();
    rank_ = 0;
    worldSize_ = 1;
  }

  int rank() const { return rank_; }
  int worldSize() const { return worldSize_; }
  bool enabled() const { return worldSize_ > 1; }
  const std::shared_ptr<tinytorch::distributed::ProcessGroup>& pg() const { return pg_; }

  void allReduceSum(tinytorch::Tensor& t) const {
    if (!enabled()) {
      return;
    }
    std::vector<tinytorch::Tensor> v{t};
    tinytorch::distributed::AllReduceOptions opts;
    opts.reduceOp = tinytorch::distributed::SUM;
    auto work = pg_->allReduce(v, opts);
    if (work) {
      work->wait();
    }
  }

  void broadcast(tinytorch::Tensor& t, int root = 0) const {
    if (!enabled()) {
      return;
    }
    std::vector<tinytorch::Tensor> v{t};
    tinytorch::distributed::BroadcastOptions opts;
    opts.rootRank = root;
    auto work = pg_->broadcast(v, opts);
    if (work) {
      work->wait();
    }
  }

 private:
  std::shared_ptr<tinytorch::distributed::ProcessGroup> pg_;
  int rank_ = 0;
  int worldSize_ = 1;
};

}  // namespace tinygpt::distributed
