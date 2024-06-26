#include "band/scheduler/shortest_expected_latency_scheduler.h"

#include <unordered_set>

#include "band/logger.h"
#include "band/time.h"

namespace band {
ShortestExpectedLatencyScheduler::ShortestExpectedLatencyScheduler(
    IEngine& engine, int window_size)
    : IScheduler(engine), window_size_(window_size) {}

bool ShortestExpectedLatencyScheduler::Schedule(JobQueue& requests) {
  bool success = true;
  JobQueue local_jobs;
  int window_size = std::min(window_size_, (int)requests.size());
  local_jobs.insert(local_jobs.begin(), requests.begin(),
                    requests.begin() + window_size);
  requests.erase(requests.begin(), requests.begin() + window_size);
  while (!local_jobs.empty()) {
    engine_.UpdateWorkersWaiting();
    // First, find the most urgent job -- the one with the
    // largest shortest latency (no, that's not a typo).
    // Put that job into some worker, and repeat this whole loop until we've
    // gone through all jobs.
    // There should be a more quicker way do this, but I'm leaving this as-is
    // to make it simple.
    // E.g., we add interpreter.GetProfiledLatency() to the expected_latency map
    // of all Jobs instead of calling GetShortestLatency() a gazillion times
    // again.

    // Note that we are NOT considering enqueue_time at the moment;
    // no request is given higher priority even if it had stayed in the queue
    // for longer than others.

    // find the most urgent job and save its index within the queue
    // 首先，找到最急迫的任务——那个具有最大最短延时的任务（是的，这并不是错别字）。
    // 将这个任务分配给一个工人，然后重复这个过程直到处理完所有任务。
    // 虽然可能有更快的方法，但为了简化操作，我决定保留这种方式。
    // 例如，我们可以添加 interpreter.GetProfiledLatency() 到所有任务的预期延时映射中，而不是反复多次调用 GetShortestLatency()。

    // 注意，我们目前不考虑任务的入队时间；
    // 即使有请求在队列中的等待时间超过其他请求，也不会得到更高的优先级。

    // 找到最紧急的任务并记录其在队列中的位置
    int64_t largest_shortest_latency = -1;
    int target_job_idx;
    SubgraphKey target_subgraph_key;
    WorkerWaitingTime worker_waiting = engine_.GetWorkerWaitingTime();

    std::unordered_set<std::pair<int, BitMask>, JobIdBitMaskHash> searched_jobs;
    for (auto it = local_jobs.begin(); it != local_jobs.end(); ++it) {
      Job& next_job = *it;

      std::pair<int, BitMask> job_to_search =
          std::make_pair(next_job.model_id, next_job.resolved_unit_subgraphs);
      if (searched_jobs.find(job_to_search) != searched_jobs.end()) {
        continue;
      } else {
        searched_jobs.insert(job_to_search);
      }

      std::pair<std::vector<SubgraphKey>, int64_t> best_subgraph =
          engine_.GetSubgraphWithShortestLatency(next_job, worker_waiting);

      if (largest_shortest_latency < best_subgraph.second) {
        largest_shortest_latency = best_subgraph.second;
        target_job_idx = it - local_jobs.begin();
        target_subgraph_key = best_subgraph.first.front();
      }
    }

    if (target_subgraph_key.IsValid() == false) {
      continue;
    }

    // for some reason, this Job must NOT be a reference (&), otherwise
    // we get a segfault at push_back() below
    // 由于某种原因，这个 Job 不能是一个引用（&），否则我们会在下面的 push_back() 处得到一个段错误
    Job most_urgent_job = local_jobs[target_job_idx];

    // remove the job from the queue so that we don't meet it in the next loop
    // 从队列中删除任务，以便在下一次循环中不再遇到它
    local_jobs.erase(local_jobs.begin() + target_job_idx);

    if (engine_.IsBegin(most_urgent_job.subgraph_key)) {
      // only set these fields if this is the first subgraph of this model
      // 如果这是该模型的第一个子图，则只设置这些字段
      most_urgent_job.expected_latency = largest_shortest_latency;
    }
    success &= engine_.EnqueueToWorker({most_urgent_job, target_subgraph_key});
  }
  return success;
}
}  // namespace band
