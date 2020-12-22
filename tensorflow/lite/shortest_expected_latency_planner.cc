#include "tensorflow/lite/shortest_expected_latency_planner.h"
#include "tensorflow/lite/profiling/time.h"
#include <iostream>

namespace tflite {
namespace impl {

void ShortestExpectedLatencyPlanner::Plan() {
  while (true) {
    if (GetSafeBool().wait())
      return;

    std::deque<Job> local_jobs;
    std::deque<Job> next_jobs;
    std::unique_lock<std::mutex> request_lock(GetRequestsMtx());
    if (!GetRequests().empty()) {
      // copy all elements to a local container so that
      // we can release the lock asap
      GetRequests().swap(local_jobs);
    } else {
      continue;
    }
    request_lock.unlock();

    while (!local_jobs.empty() || !next_jobs.empty()) {
      if (local_jobs.empty())
        local_jobs.swap(next_jobs);

      // First, find the most urgent job -- the one with the
      // largest shortest latency (no, that's not a typo).
      // Put that job into some worker, and repeat this whole loop until we've
      // gone through all jobs.
      // There should be a more quicker way do this, but I'm leaving this as-is
      // to make it simple.
      // E.g., we add subgraph.GetExpectedLatency() to the expected_latency map
      // of all Jobs instead of calling GetShortestLatency() a gazillion times
      // again.

      // Note that we are NOT considering enqueue_time at the moment;
      // no request is given higher priority even if it had stayed in the queue
      // for longer than others.

      bool is_latency_critical = false;
      int target_idx;
      TfLiteDevice target_device;
      Job latency_critical_job = Job(-1);
      for (auto it = local_jobs.begin(); it != local_jobs.end(); ++it) {
        Job& to_execute = *it;
        if (to_execute.slo_ms_ > 0) {
          is_latency_critical = true;
          latency_critical_job = to_execute;
          target_idx = it - local_jobs.begin();
          local_jobs.erase(local_jobs.begin() + target_idx);
          break;
        }
      }
      
      if (is_latency_critical) {
        for (int i = 0; i < GetInterpreter()->GetWorkersSize(); ++i) {
          Worker& worker = GetInterpreter()->GetWorker(i);
          {
            std::lock_guard<std::mutex> lock(worker.GetDeviceMtx());
            std::vector<int> to_erase;
            for (auto it = worker.GetDeviceRequests().begin(); it != worker.GetDeviceRequests().end(); ++it) {
              Job current = *it;
              if (current.invoke_time_ == 0 && current.slo_ms_ == 0) {
                int idx = it - worker.GetDeviceRequests().begin();
                next_jobs.push_back(current);
                to_erase.push_back(idx);
              }
            }

            for (int k = to_erase.size() - 1; k >= 0; --k) {
              worker.GetDeviceRequests().erase(worker.GetDeviceRequests().begin() + to_erase[k]);
            }
          }
          // std::cout << "next job size : " << next_jobs.size() << std::endl;
        }

        local_jobs.swap(next_jobs);
        local_jobs.push_front(latency_critical_job);

        auto it = local_jobs.begin();
        Job& to_execute = *it;
        target_idx = it - local_jobs.begin();
        target_device = GetInterpreter()->GetShortestLatency(to_execute.model_id_, to_execute);

        // std::cout << "MODEL : " << to_execute.model_id_ << std::endl;
        // std::cout << "device : " << target_device << std::endl;
      } else {
        // find the most urgent job and save its index within the queue
        int64_t largest_shortest_latency = -1;
        for (auto it = local_jobs.begin(); it != local_jobs.end(); ++it) {
          Job& to_execute = *it;
          TfLiteDevice device = GetInterpreter()->GetShortestLatency(to_execute.model_id_, to_execute);
          int64_t shortest_latency = to_execute.expected_latency[device];

          if (shortest_latency > largest_shortest_latency) {
            largest_shortest_latency = shortest_latency;
            target_idx = it - local_jobs.begin();
            target_device = device;
          }
        }
      }

      // for some reason, this Job must NOT be a reference (&), otherwise
      // we get a segfault at push_back() below
      Job most_urgent_job = local_jobs[target_idx];

      // remove the job from the queue so that we don't meet it in the next loop
      local_jobs.erase(local_jobs.begin() + target_idx);
      Worker& worker = GetInterpreter()->GetWorker(target_device);

      {
        std::lock_guard<std::mutex> lock(worker.GetDeviceMtx());
        int subgraph_idx = GetInterpreter()->GetSubgraphIdx(most_urgent_job.model_id_, target_device);
        most_urgent_job.subgraph_idx_ = subgraph_idx;
        most_urgent_job.device_id_ = target_device;

        worker.GetDeviceRequests().push_back(most_urgent_job);
        worker.GetRequestCv().notify_one();
      }
    }
  }
}

}  // namespace impl
}  // namespace tflite
