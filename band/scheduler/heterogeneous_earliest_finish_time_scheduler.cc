#include "band/scheduler/heterogeneous_earliest_finish_time_scheduler.h"

#include <unordered_set>

#include "band/device/thermal.h"
#include "band/job_tracer.h"
#include "band/logger.h"

namespace band {
HEFTScheduler::HEFTScheduler(IEngine& engine, int window_size)
    : IScheduler(engine), window_size_(window_size) {}

bool HEFTScheduler::Schedule(JobQueue& requests) {
  BAND_TRACER_SCOPED_THREAD_EVENT(ScheduleFunction);
  bool success = true;
  auto thermal = ThermalMap();
  thermal[SensorFlag::kCPU] = 25.f;
  thermal[SensorFlag::kGPU] = 25.f;
  thermal[SensorFlag::kDSP] = 25.f;
  thermal[SensorFlag::kNPU] = 25.f;
  thermal[SensorFlag::kTarget] = 25.f;

  int num_jobs = std::min(window_size_, (int)requests.size());
  while (num_jobs > 0) {
    BAND_TRACER_SCOPED_THREAD_EVENT(ScheduleJob);
    engine_.UpdateWorkersWaiting();

    // stop if there are no idle devices.
    std::set<int> idle_workers = engine_.GetIdleWorkers();
    if (idle_workers.empty()) {
      break;
    }

    // hold on to a local copy of worker waiting time
    WorkerWaitingTime waiting_time = engine_.GetWorkerWaitingTime();
    std::set<JobId> jobs_to_yield;

    double largest_min_cost = -1;
    int target_job_index;
    SubgraphKey target_subgraph_key;
    SubgraphKey target_subgraph_key_next;

    do {
      BAND_TRACER_SCOPED_THREAD_EVENT(ScheduleWhileLoop);
      largest_min_cost = -1;
      target_job_index = -1;

      // only check up to `num_jobs` requests
      std::unordered_set<std::pair<int, BitMask>, JobIdBitMaskHash>
          searched_jobs;
      for (auto it = requests.begin(); it != requests.begin() + num_jobs;
           ++it) {
        BAND_TRACER_SCOPED_THREAD_EVENT(ScheduleForLoop);
        Job job = *it;

        if (jobs_to_yield.find(job.job_id) != jobs_to_yield.end()) {
          continue;
        }

        std::pair<ModelId, BitMask> job_to_search =
            std::make_pair(job.model_id, job.resolved_unit_subgraphs);
        if (searched_jobs.find(job_to_search) != searched_jobs.end()) {
          continue;
        }

        searched_jobs.insert(job_to_search);
        {
          BAND_TRACER_SCOPED_THREAD_EVENT(BestSubgraph);
          const auto& best_subgraph = engine_.GetSubgraphWithMinCost(
              job, waiting_time, thermal,
              [](double lat, const std::map<SensorFlag, double>&) -> double {
                return lat;
              });
          BAND_LOG_PROD(BAND_LOG_INFO, "cost: %f",
                        std::get<2>(best_subgraph.second));
          if (largest_min_cost < std::get<2>(best_subgraph.second)) {
            largest_min_cost = std::get<2>(best_subgraph.second);
            target_subgraph_key = best_subgraph.first.front();
            target_job_index = it - requests.begin();
            if (best_subgraph.first.size() > 1) {
              target_subgraph_key_next = best_subgraph.first[1];
            } else {
              target_subgraph_key_next = {};
            }
          }
        }
      }

      // no one wants to be scheduled.
      if (target_job_index < 0) {
        return success;
      }

      // skip this job if we can't schedule it immediately,
      // even if this job is the "most urgent" one
      const int worker_id = target_subgraph_key.GetWorkerId();
      if (idle_workers.find(worker_id) != idle_workers.end()) {
        break;
      }
      waiting_time[worker_id] += engine_.GetExpected(target_subgraph_key);
      auto requests_it = requests.begin() + target_job_index;
      Job& job = *requests_it;

      jobs_to_yield.insert(job.job_id);
    } while (true);

    auto requests_it = requests.begin() + target_job_index;
    Job job = *requests_it;

    // erase the job from requests and decrement num_jobs
    requests.erase(requests_it);
    num_jobs--;

    if (engine_.IsBegin(target_subgraph_key)) {
      // only set these fields if this is the first subgraph of this model
      job.cost = largest_min_cost;
    }

    success &= engine_.EnqueueToWorker({job, target_subgraph_key});
  }

  return success;
}
}  // namespace band