#include "band/scheduler/heterogeneous_earliest_finish_time_scheduler.h"

#include <unordered_set>

#include "band/logger.h"

namespace band {
HEFTScheduler::HEFTScheduler(IEngine& engine, int window_size, bool reserve)
    : IScheduler(engine), window_size_(window_size), reserve_(reserve) {}

bool HEFTScheduler::Schedule(JobQueue& requests) {
  bool success = true;
  int window_size = std::min(window_size_, (int)requests.size());
  // stop if there are no idle devices OR there's nothing in `requests`
  // 如果没有空闲设备或 `requests` 中为空则停止
  while (window_size > 0) {
    engine_.UpdateWorkersWaiting();
    std::set<int> idle_workers = engine_.GetIdleWorkers();
    if (idle_workers.empty()) {
      break;
    }

    // hold on to a local copy of worker waiting time
    // 保存一个关于工作器等待时间的本地副本
    WorkerWaitingTime waiting_time = engine_.GetWorkerWaitingTime();
    std::set<JobId> jobs_to_yield;
    // basically the same as ShortestExpectedLatencyScheduler
    // 本质上与 ShortestExpectedLatencyScheduler 相同
    int64_t largest_shortest_latency;
    int64_t target_job_index;
    SubgraphKey target_subgraph_key;
    SubgraphKey target_subgraph_key_next;
    do {
      largest_shortest_latency = -1;
      target_job_index = -1;

      // only check up to `window_size` requests
      std::unordered_set<std::pair<int, BitMask>, JobIdBitMaskHash>
          searched_jobs;
      for (auto it = requests.begin(); it != requests.begin() + window_size;
           ++it) {
            // 循环遍历请求队列中的作业，但只考虑前 window_size 个作业
        Job job = *it;

        if (jobs_to_yield.find(job.job_id) != jobs_to_yield.end()) {
          // 首先检查它是否已经被标记为不可立即调度
          continue;
        }

        // 检查是否已经评估过相同的模型ID和已解决单元子图组合
        std::pair<int, BitMask> job_to_search =
            std::make_pair(job.model_id, job.resolved_unit_subgraphs);
        if (searched_jobs.find(job_to_search) != searched_jobs.end()) {
          continue;
        } else {
          searched_jobs.insert(job_to_search);
        }

        // update waiting_time for all future jobs in reserved_
        // 对于已预留的作业，更新其在特定工作器上的等待时间
        // 这是通过叠加所有预留作业在该工作器上预期完成时间来实现的。
        WorkerWaitingTime reserved_time(waiting_time);
        for (auto job_subgraph_key : reserved_) {
          if (job_subgraph_key.first == job.job_id) {
            continue;
          }

          reserved_time[job_subgraph_key.second.GetWorkerId()] +=
              engine_.GetExpected(job_subgraph_key.second);
        }

        std::pair<std::vector<SubgraphKey>, int64_t> best_subgraph =
            engine_.GetSubgraphWithShortestLatency(job, reserved_time);

        if (largest_shortest_latency < best_subgraph.second) {
          largest_shortest_latency = best_subgraph.second;
          target_subgraph_key = best_subgraph.first.front();
          target_job_index = it - requests.begin();
          if (best_subgraph.first.size() > 1) {
            target_subgraph_key_next = best_subgraph.first[1];
          } else {
            target_subgraph_key_next = {};
          }
        }
      }

      if (target_job_index < 0) {
        // no one wants to be scheduled..
        return success;
      }

      // skip this job if we can't schedule it immediately,
      // even if this job is the "most urgent" one
      // 即使这个作业是“最紧急”的，如果我们不能立即调度它，我们也会跳过这个作业
      const int worker_id = target_subgraph_key.GetWorkerId();
      if (idle_workers.find(worker_id) == idle_workers.end()) {
        // 没有找到空闲的工作器
        waiting_time[worker_id] += engine_.GetExpected(target_subgraph_key);
        auto requests_it = requests.begin() + target_job_index;
        Job job = *requests_it;
        jobs_to_yield.insert(job.job_id);
        // 不可立即调度
        continue;
      } else {
        break;
      }
    } while (true);

    auto requests_it = requests.begin() + target_job_index;
    Job job = *requests_it;

    // erase the job from requests and decrement window_size
    // 从请求列表中移除该任务并减少窗口大小
    requests.erase(requests_it);
    window_size--;

    // Update Job status specific to this planner.
    // Common status will be updated by `EnqueueAction`.
    // 更新特定于此调度器的作业状态。
    // 通用状态将由 `EnqueueAction` 更新。
    if (engine_.IsBegin(target_subgraph_key)) {
      // only set these fields if this is the first subgraph of this model
      // 仅在这是该模型的第一个子图时才设置这些字段
      job.expected_latency = largest_shortest_latency;
    }

    success &= engine_.EnqueueToWorker({job, target_subgraph_key});

    if (reserve_) {
      // add next job to reserved_, if one exists
      // 调度器会在调度作业时考虑将某些资源或计算时间预留给特定的作业，以确保这些作业可以按计划执行。
      if (target_subgraph_key_next != SubgraphKey()) {
        // 如果有后续子图
        reserved_[job.job_id] = target_subgraph_key_next;
      } else {
        reserved_.erase(job.job_id);
        // 如果没有后续子图，则从预留列表中删除该作业
      }
    }
  }
  return success;
}
}  // namespace band