#ifndef BAND_SCHEDULER_FIXED_WORKER_SCHEDULER_H_
#define BAND_SCHEDULER_FIXED_WORKER_SCHEDULER_H_

#include "band/scheduler/scheduler.h"

namespace band {

// Assigns requested model to devices according to a direct request from engine
// or model_id.
// 分配请求的模型到设备，根据来自引擎的直接请求或模型id
class FixedWorkerScheduler : public IScheduler {
  // 固定工作器的调度策略，通常用于将作业调度到特定资源上
 public:
  using IScheduler::IScheduler;
  bool Schedule(JobQueue& requests) override;
  bool NeedFallbackSubgraphs() override { return false; }
  WorkerType GetWorkerType() override { return WorkerType::kDeviceQueue; }
};

class FixedWorkerGlobalQueueScheduler : public IScheduler {
  // 全局队列的调度策略，适用于需要跨多个工作器或设备统一管理和调度作业的场景
 public:
  using IScheduler::IScheduler;
  bool Schedule(JobQueue& requests) override;
  // Required for checking SLO violation.
  // We could add an option to this planner for skipping the SLO check,
  // in which case this function can return false.
  // 用于检查SLO违规，我们可以为此计划程序添加一个选项，以跳过SLO检查，在这种情况下，此函数可以返回false
  bool NeedFallbackSubgraphs() override { return false; }
  WorkerType GetWorkerType() override { return WorkerType::kGlobalQueue; }
};

}  // namespace band

#endif  // BAND_SCHEDULER_fixed_worker_scheduler_H_
