#include "tensorflow/lite/worker.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"
#include <iostream>

namespace tflite {
namespace impl {

Worker::Worker(std::shared_ptr<Planner> planner, int device_idx)
  : device_cpu_thread_([this] { this->Work(); }), device_idx_(device_idx) {
  planner_ = planner;
}

Worker::~Worker() {
  {
    std::lock_guard<std::mutex> lock(device_mtx_);
    kill_worker_ = true;
  }
  request_cv_.notify_all();
  device_cpu_thread_.join();
}

TfLiteStatus Worker::SetWorkerThreadAffinity(const CpuSet& thread_affinity_mask) {
  cpu_set_ = thread_affinity_mask;
  return SetCPUThreadAffinity(cpu_set_);
}

void Worker::Work() {
  while (true) {
    std::unique_lock<std::mutex> lock(device_mtx_);
    request_cv_.wait(lock, [this]() {
      return kill_worker_ || !this->requests_.empty();
    });

    if (requests_.empty()) {
      lock.unlock();
      break;
    }

    Job& job = requests_.front();
    // requests_.pop_front();
    // this is techincally not the correct invoke_time, but
    // just record it now to avoid acquiring the lock again later
    job.invoke_time_ = profiling::time::NowMicros();
    lock.unlock();

    int subgraph_idx = job.subgraph_idx_;
    std::shared_ptr<Planner> planner_ptr = planner_.lock();
    if (planner_ptr) {
      Interpreter* interpreter_ptr = planner_ptr->GetInterpreter();
      Subgraph& subgraph = *(interpreter_ptr->subgraph(subgraph_idx));

      if (subgraph.Invoke() == kTfLiteOk) {
        job.end_time_ = profiling::time::NowMicros();
        planner_ptr->EnqueueFinishedJob(job);
      } else {
        // TODO #21: Handle errors in multi-thread environment
        // Currently, put a job with a minus sign if Invoke() fails.
        job.end_time_ = profiling::time::NowMicros();
        planner_ptr->EnqueueFinishedJob(Job(-1 * subgraph_idx));
      }

      lock.lock();
      requests_.pop_front();
      bool empty = requests_.empty();
      lock.unlock();

      if (empty) {
        TryWorkSteal();
      }

      planner_ptr->GetSafeBool().notify();
    } else {
      // TODO #21: Handle errors in multi-thread environment
      return;
    }
  }
}

void Worker::TryWorkSteal() {
  std::shared_ptr<Planner> planner_ptr = planner_.lock();
  if (!planner_ptr) {
    std::cout << "Worker " << device_idx_
              << " TryWorkSteal() Failed to acquire pointer to Planner"
              << std::endl;
    return;
  }

  Interpreter* interpreter_ptr = planner_ptr->GetInterpreter();
  int64_t max_latency_gain = -1;
  int max_latency_gain_device = -1;
  for (int i = 0; i < interpreter_ptr->GetWorkersSize(); ++i) {
    if (i == device_idx_) {
      continue;
    }

    Worker& worker = interpreter_ptr->GetWorker(i);
    int64_t waiting_time = worker.GetWaitingTime();

    std::unique_lock<std::mutex> lock(worker.GetDeviceMtx());
    if (worker.GetDeviceRequests().empty()) {
      // there's nothing to steal here
      continue;
    }

    Job& job = worker.GetDeviceRequests().back();
    if (job.invoke_time_ > 0) {
      // this job is being processed by the target worker, so leave it alone
      // FIXME: assume that invoke_time_ is updated only while
      // the lock is acquired
      continue;
    }
    lock.unlock();

    int subgraph_idx = interpreter_ptr->GetSubgraphIdx(
        job.model_id_, static_cast<TfLiteDevice>(device_idx_));
    Subgraph* subgraph = interpreter_ptr->subgraph(subgraph_idx);
    if (!subgraph) {
      // a subgraph for this model on this device isn't available
      continue;
    }

    int64_t expected_latency = subgraph->GetExpectedLatency();
    if (expected_latency > waiting_time) {
      // no point in stealing this job, it's just going to take longer
      continue;
    }

    int64_t latency_gain = waiting_time - expected_latency;
    if (latency_gain > max_latency_gain) {
      max_latency_gain = latency_gain;
      max_latency_gain_device = i;
    }
  }


  if (max_latency_gain < 0) {
    // no viable job to steal -- do nothing
    return;
  }

  Worker& worker = interpreter_ptr->GetWorker(max_latency_gain_device);
  std::unique_lock<std::mutex> lock(worker.GetDeviceMtx(), std::defer_lock);
  std::unique_lock<std::mutex> my_lock(device_mtx_, std::defer_lock);
  std::lock(lock, my_lock);

  if (worker.GetDeviceRequests().empty()) {
    // target worker has went on and finished all of its jobs
    // while we were slacking off
    return;
  }

  // this must not be a reference,
  // otherwise the pop_back() below will invalidate it
  Job job = worker.GetDeviceRequests().back();
  if (job.invoke_time_ > 0) {
    // make sure the target worker hasn't started processing the job yet
    return;
  }

  if (!requests_.empty()) {
    // make sure that I still don't have any work to do
    return;
  }

  // std::cout << "Worker " << device_idx_ << " is stealing from "
  //           << "worker " << max_latency_gain_device << std::endl;

  int subgraph_idx = interpreter_ptr->GetSubgraphIdx(
      job.model_id_, static_cast<TfLiteDevice>(device_idx_));
  job.subgraph_idx_ = subgraph_idx;
  job.device_id_ = device_idx_;

  // finally, we perform the swap
  worker.GetDeviceRequests().pop_back();
  requests_.push_back(job);
}

int64_t Worker::GetWaitingTime() {
  std::unique_lock<std::mutex> lock(device_mtx_);

  int64_t total = 0;
  for (std::deque<Job>::iterator it = requests_.begin(); it != requests_.end(); ++it) {
    int subgraph_idx = (*it).subgraph_idx_;
    std::shared_ptr<Planner> planner_ptr = planner_.lock();
    if (planner_ptr) {
      Interpreter* interpreter_ptr = planner_ptr->GetInterpreter();
      Subgraph& subgraph = *(interpreter_ptr->subgraph(subgraph_idx));
      int64_t subgraph_latency = subgraph.GetExpectedLatency();
      total += subgraph_latency;

      if (it == requests_.begin()) {
        int64_t current_time = profiling::time::NowMicros();
        int64_t invoke_time = (*it).invoke_time_;
        if (invoke_time > 0 && current_time > invoke_time) {
          int64_t progress = (current_time - invoke_time) > subgraph_latency ? subgraph_latency : (current_time - invoke_time);
          total -= progress;
          // std::cout << "Invoke Time : " << (*it).invoke_time_ << std::endl;
          // std::cout << "current Time : " << current_time << std::endl;
          // std::cout << "subgraph_latency : " << subgraph_latency << std::endl;
          // std::cout << "progress : " << progress << std::endl;
        }
      }
    } else {
      return -1;
    }
  }
  lock.unlock();

  return total;
}

}  // namespace impl
}  // namespace tflite
