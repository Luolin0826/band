#include "tensorflow/lite/shortest_expected_latency_planner.h"
#include "tensorflow/lite/profiling/time.h"
#include <iostream>

namespace tflite {
namespace impl {

void ShortestExpectedLatencyPlanner::Plan() {
  while (true) {
    if (GetSafeBool().wait())
      return;

    std::vector<bool> is_worker_empty;
    is_worker_empty.resize(GetInterpreter()->GetWorkersSize(), false);

    for (int i = 0; i < GetInterpreter()->GetWorkersSize(); ++i) {
      Worker& worker = GetInterpreter()->GetWorker(i);
      {
        std::lock_guard<std::mutex> lock(worker.GetDeviceMtx());
        if (worker.GetDeviceRequests().empty()) {
          is_worker_empty[i] = true;
        }
      }
    }
 
    // std::cout << std::endl;
    // std::cout << "START PLAN!" << std::endl;
    std::unique_lock<std::mutex> request_lock(GetRequestsMtx());
    // while (!GetRequests().empty()) {
    for (std::deque<Job>::iterator it = GetRequests().begin(); it != GetRequests().end(); ++it) {
      bool continue_plan = false;
      for (int i = 0; i < is_worker_empty.size(); ++i) {
        continue_plan |= is_worker_empty[i];
      }

      if (!continue_plan) {
        break;
      }

      // Job to_execute = GetRequests().front();
      // GetRequests().pop_front();
      Job to_execute = *it;
      int model_id = to_execute.model_id_;
      TfLiteDevice device = GetInterpreter()->GetShortestLatency(model_id);

      if (is_worker_empty[device]) {
        Worker& worker = GetInterpreter()->GetWorker(device);
        {
          std::lock_guard<std::mutex> lock(worker.GetDeviceMtx());
          int subgraph_idx = GetInterpreter()->GetSubgraphIdx(model_id, device);
          to_execute.subgraph_idx_ = subgraph_idx;
          to_execute.device_id_ = device;
          GetRequests().erase(it);

          worker.GetDeviceRequests().push_back(to_execute);
          worker.GetRequestCv().notify_one();

          is_worker_empty[device] = false;
        }
      }
    }
    request_lock.unlock();
  }
}

}  // namespace impl
}  // namespace tflite
