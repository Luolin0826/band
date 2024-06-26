#include "band/scheduler/fixed_worker_scheduler.h"

namespace band {
bool FixedWorkerScheduler::Schedule(JobQueue& requests) {
  bool success = true;
  // TODO: fallback subgraphs for FixedDeviceFixedWorkerPlanner?
  // 待办：为 FixedDeviceFixedWorkerPlanner 设计回退子图方案？
  while (!requests.empty()) {
    Job to_execute = requests.front();
    requests.pop_front();  // erase job

    int model_id = to_execute.model_id;
    // Priority
    // (1) : direct request from the engine
    // (2) : predefined mapping from the config
    // 优先级
    // (1) : 直接来自引擎的请求
    // (2) : 根据配置预设的映射关系
    WorkerId worker_id = to_execute.target_worker_id == -1
                             ? engine_.GetModelWorker(model_id)
                             : to_execute.target_worker_id;
    SubgraphKey key = engine_.GetLargestSubgraphKey(model_id, worker_id);
    success &= engine_.EnqueueToWorker({to_execute, key});
  }
  return success;
}

}  // namespace band
