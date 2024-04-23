#ifndef BAND_SCHEDULER_LEAST_SLACK_FIRST_SCHEDULER_H_
#define BAND_SCHEDULER_LEAST_SLACK_FIRST_SCHEDULER_H_

#include "band/scheduler/scheduler.h"

namespace band {

class LeastSlackFirstScheduler : public IScheduler {
 public:
  explicit LeastSlackFirstScheduler(IEngine& engine, int window_size);

  bool Schedule(JobQueue& requests) override;
  bool NeedFallbackSubgraphs() override { return true; }
  WorkerType GetWorkerType() override { return WorkerType::kGlobalQueue; }

 private:
  int64_t GetSlackTime(int64_t current_time, const Job& job);
  // 计算给定作业的松弛时间，即任务的最后期限与当前时间和预期剩余执行时间之差。
  void SortBySlackTime(JobQueue& requests, int window_size,
                       int64_t current_time);
                      //  根据作业的松弛时间对请求队列进行排序
  void UpdateExpectedLatency(JobQueue& requests, int window_size);
  // 更新队列中作业的预期延迟，这通常是基于最短延迟的子图计算得出。
  const int window_size_;
};

}  // namespace band

#endif  // BAND_SCHEDULER_LEAST_SLACK_FIRST_SCHEDULER_H_
