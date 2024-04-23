#ifndef BAND_SCHEDULER_HETEROGENEOUS_EARLIEST_FINISH_TIME_SCHEDULER_H_
#define BAND_SCHEDULER_HETEROGENEOUS_EARLIEST_FINISH_TIME_SCHEDULER_H_

#include "band/scheduler/scheduler.h"
// HEFT 算法主要考虑每个作业在不同工作器上的预计完成时间，并尝试将作业调度到可以最早完成该作业的工作器上
namespace band {

class HEFTScheduler : public IScheduler {
 public:
  explicit HEFTScheduler(IEngine& engine, int window_size, bool reserve);
  // window_size 调度窗口大小 控制决策调度的范围和精度
  // reserve：一个布尔值，指示是否启用资源预留功能，这可以为关键作业预留计算资源，保证其按时完成

  bool Schedule(JobQueue& requests) override;
  bool NeedFallbackSubgraphs() override { return true; }
  WorkerType GetWorkerType() override { return WorkerType::kGlobalQueue; }

 private:
  // job_id --> subgraph_key
  std::map<int, SubgraphKey> reserved_;
  const int window_size_;
  const bool reserve_;
};

}  // namespace band

#endif  // BAND_SCHEDULER_HETEROGENEOUS_EARLIEST_FINISH_TIME_SCHEDULER_H_
