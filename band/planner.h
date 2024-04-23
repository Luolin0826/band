#ifndef BAND_PLANNER_H_
#define BAND_PLANNER_H_

#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "band/config.h"
#include "band/safe_bool.h"
#include "band/scheduler/scheduler.h"
#include "band/worker.h"

namespace band {

// The maximum number of available job outputs at one time.
// 在一次性的作业输出中可用的最大数量
#define NUM_FINISHED_RECORDS 1000

// The job queue which can be shared by multiple threads.
// 可由多个线程共享的作业队列
struct ConcurrentJobQueue {
  JobQueue queue;
  std::mutex mtx;
};

class Planner {
 public:
  explicit Planner(IEngine& engine);
  ~Planner();

  absl::Status Init(const PlannerConfig& config);
  absl::Status AddScheduler(std::unique_ptr<IScheduler> scheduler);

  // Enqueues a job to a worker request queue.
  // 通过将作业排队到工作请求队列来排队作业
  JobId EnqueueRequest(Job job, bool push_front = false);

  // Enqueues a batch of jobs to a worker request queue.
  // Assigns new job id for non-continuous job.
  // 入队一批作业到工作请求队列
  // 为非连续作业分配新的作业id
  std::vector<JobId> EnqueueBatch(std::vector<Job> jobs,
                                  bool push_front = false);
  // Waits until the jobs are done.
  // The interpreter calls the method.
  // 等待作业完成
  // 解释器调用该方法
  void Wait(std::vector<int> job_ids);
  void WaitAll();
  // Enqueues a finised job to the queue.
  // A worker calls the method.
  // 将完成的作业排队到队列
  // 工作器调用该方法
  void EnqueueFinishedJob(Job& job);
  void PrepareReenqueue(Job& job);

  // Enqueue the request to the worker.
  // Returns true if the request is successfully enqueued.
  // 将调度动作加入到具体的工作器
  bool EnqueueToWorker(const std::vector<ScheduleAction>& action);
  void Trigger() { planner_safe_bool_.notify(); }

  // Checks if the schedulers can handle fallback subgraphs.
  // Returns true if any of the scheduler can handle fallback subgraphs.
  // But, note that having both types of scheduler (w/ fallback, w/o fallback),
  // may lead to unexpected results.
  // 此功能用于检查调度器是否能够管理回退子图。
  // 如果至少有一个调度器能处理回退子图，就返回 true。
  // 但需要注意的是，如果同时使用带回退和不带回退的调度器，可能会引起一些意外的问题。
  bool NeedFallbackSubgraphs() const;

  std::mutex& GetRequestsMtx() { return requests_.mtx; }
  JobQueue& GetRequests() { return requests_.queue; }
  int GetWindowSize() const { return schedule_window_size_; }
  void SetWindowSize(int schedule_window_size);
  const std::map<int, int>& GetModelExecutionCounts() const {
    return model_execution_count_;
  }
  // Sets the callback function pointer to report the end of invoke.
  // 设置一个回调函数，当作业结束时调用
  CallbackId SetOnEndRequest(
      std::function<void(int, absl::Status)> on_end_request);
  absl::Status UnsetOnEndRequest(CallbackId callback_id);

  // Get the Job instance with the `job_id`.
  // 获取已经完成的作业
  Job GetFinishedJob(int job_id);
  // Get which worker types the schedulers require.
  // 获取调度器需要的工作器类型
  int GetWorkerType() const;
  // 获取模型与工作器的映射关系
  std::map<ModelId, WorkerId>& GetModelWorkerMap() { return model_worker_map_; }

 private:
  // Main loop for planner_thread_
  // 主调度循环，负责持续调度作业
  absl::Status Plan();
  // Write job logs and delete the job from the finished queue.
  // 清理已完成的作业记录记录
  void FlushFinishedJobs();
  // Copy the Job instances from the `requests_` to the local queue.
  // Note that this function is to minimize the hold time for the queue lock.
  // 将请求队列中的作业复制到本地队列，减少锁持有时间
  void CopyToLocalQueues();
  // 检查是否违反了指定的SLO
  // Check if the job violated the specified SLO.
  // This func assumes that workers_waiting_, job.profiled_time,
  // job.device_id, and job.enqueue_time are all up to date.
  bool IsSLOViolated(Job& job);
  // Update the job information based on next target key
  // 更新作业的调度状态
  void UpdateJobScheduleStatus(Job& job, const SubgraphKey& target_key);
  // Update `model_worker_map_`.
  void TryUpdateModelWorkerMapping();
  bool IsJobIdValid(int job_id);
  int GetJobRecordIndex(int job_id) const;

  CpuSet cpu_set_;
  bool need_cpu_update_ = false;

  SafeBool planner_safe_bool_;

  // Jobs Finished
  std::map<int, int> model_execution_count_;

  mutable std::mutex on_end_request_mtx_;
  std::map<CallbackId, std::function<void(int, absl::Status)>>
      on_end_request_callbacks_;
  CallbackId next_callback_id_ = 0;

  // Request Queue
  ConcurrentJobQueue requests_;

  // Multi-level Local Queue.
  // The closer the index is to 0, the higher the priority.
  std::vector<JobQueue> local_queues_;
  std::vector<std::unique_ptr<IScheduler>> schedulers_;

  std::mutex job_finished_mtx_;
  std::array<Job, NUM_FINISHED_RECORDS> jobs_finished_record_;
  std::atomic<int> num_submitted_jobs_;
  int num_finished_jobs_ = 0;

  std::condition_variable end_invoke_;
  std::string log_path_;

  int schedule_window_size_ = std::numeric_limits<int>::max();

  std::thread planner_thread_;
  // Map structure to find assigned worker of model idx (model_id, worker_id)
  std::map<ModelId, WorkerId> model_worker_map_;
  IEngine& engine_;
};

}  // namespace band

#endif  // BAND_PLANNER_H_
