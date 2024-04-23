#ifndef BAND_SCHEDULER_SCHEDULER_H_
#define BAND_SCHEDULER_SCHEDULER_H_

#include <map>

#include "band/engine_interface.h"

namespace band {
class Planner;

class IScheduler {
 public:
  explicit IScheduler(IEngine& engine) : engine_(engine) {}
  virtual ~IScheduler() = default;
  // A Schedule() function is expected to do the followings:
  // For the given requests, selected requests to schedule and
  // find the appropriate devices. The selected requests should be
  // enqueued to the worker and removed from original queue.
  // Returns false if the scheduler wants to be called again.
  // Schedule() 函数的预期功能如下：
  // 针对给定的请求，挑选出需要调度的请求并寻找合适的设备。选中的请求应该被加入工作人员的队列并从原始队列中删除。
  // 如果调度器需要再次执行，则返回 false。
  virtual bool Schedule(JobQueue& requests) = 0;
  virtual bool NeedFallbackSubgraphs() = 0;
  virtual WorkerType GetWorkerType() = 0;

 protected:
  IEngine& engine_;
};
}  // namespace band

#endif