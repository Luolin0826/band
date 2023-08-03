#ifndef BAND_CONFIG_BUILDER_H_
#define BAND_CONFIG_BUILDER_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "band/common.h"
#include "band/config.h"
#include "band/device/cpu.h"
#include "band/error_reporter.h"

namespace band {

class ProfileConfigBuilder {
  friend class RuntimeConfigBuilder;  // TODO: Find a safer way for
                                      // RuntimeConfigBuilder to access
                                      // variables
 public:
  ProfileConfigBuilder() {
    copy_computation_ratio_ = std::vector<int>(EnumLength<DeviceFlag>(), 30000);
  }
  ProfileConfigBuilder& AddOnline(bool online) {
    online_ = online;
    return *this;
  }
  ProfileConfigBuilder& AddNumWarmups(int num_warmups) {
    num_warmups_ = num_warmups;
    return *this;
  }
  ProfileConfigBuilder& AddNumRuns(int num_runs) {
    num_runs_ = num_runs;
    return *this;
  }
  ProfileConfigBuilder& AddCopyComputationRatio(
      std::vector<int> copy_computation_ratio) {
    copy_computation_ratio_ = copy_computation_ratio;
    return *this;
  }
  ProfileConfigBuilder& AddProfileDataPath(std::string profile_data_path) {
    profile_data_path_ = profile_data_path;
    return *this;
  }
  ProfileConfigBuilder& AddSmoothingFactor(float smoothing_factor) {
    smoothing_factor_ = smoothing_factor;
    return *this;
  }

  absl::StatusOr<ProfileConfig> Build();
  absl::Status IsValid();

 private:
  bool online_ = true;
  int num_warmups_ = 1;
  int num_runs_ = 1;
  std::vector<int> copy_computation_ratio_;
  std::string profile_data_path_ = "";
  float smoothing_factor_ = 0.1;
};

// Builder for creating PlannerConfig
class PlannerConfigBuilder {
  friend class RuntimeConfigBuilder;

 public:
  PlannerConfigBuilder& AddScheduleWindowSize(int schedule_window_size) {
    schedule_window_size_ = schedule_window_size;
    return *this;
  }
  PlannerConfigBuilder& AddSchedulers(std::vector<SchedulerType> schedulers) {
    if (schedulers.size() != 0) {
      schedulers_ = schedulers;
    }
    return *this;
  }
  PlannerConfigBuilder& AddCPUMask(CPUMaskFlag cpu_mask) {
    cpu_mask_ = cpu_mask;
    return *this;
  }
  PlannerConfigBuilder& AddLogPath(std::string log_path) {
    log_path_ = log_path;
    return *this;
  }

  absl::StatusOr<PlannerConfig> Build();

 private:
  absl::Status IsValid();

  int schedule_window_size_ = std::numeric_limits<int>::max();
  std::vector<SchedulerType> schedulers_;
  CPUMaskFlag cpu_mask_ = CPUMaskFlag::kAll;
  std::string log_path_ = "";
};

// Builder for creating WorkerConfig.
class WorkerConfigBuilder {
  friend class RuntimeConfigBuilder;

 public:
  WorkerConfigBuilder() {
    for (size_t i = 0; i < EnumLength<DeviceFlag>(); i++) {
      workers_.push_back(static_cast<DeviceFlag>(i));
    }
    cpu_masks_ =
        std::vector<CPUMaskFlag>(EnumLength<DeviceFlag>(), CPUMaskFlag::kAll);
    num_threads_ = std::vector<int>(EnumLength<DeviceFlag>(), 1);
  }
  WorkerConfigBuilder& AddWorkers(std::vector<DeviceFlag> workers) {
    if (workers.size() != 0) {
      workers_ = workers;
    }
    return *this;
  }
  WorkerConfigBuilder& AddCPUMasks(std::vector<CPUMaskFlag> cpu_masks) {
    if (cpu_masks.size() != 0) {
      cpu_masks_ = cpu_masks;
    }
    return *this;
  }
  WorkerConfigBuilder& AddNumThreads(std::vector<int> num_threads) {
    if (num_threads.size() != 0) {
      num_threads_ = num_threads;
    }
    return *this;
  }
  WorkerConfigBuilder& AddAllowWorkSteal(bool allow_worksteal) {
    allow_worksteal_ = allow_worksteal;
    return *this;
  }
  WorkerConfigBuilder& AddAvailabilityCheckIntervalMs(
      int availability_check_interval_ms) {
    availability_check_interval_ms_ = availability_check_interval_ms;
    return *this;
  }
  absl::StatusOr<WorkerConfig> Build();

 private:
  absl::Status IsValid();

  std::vector<DeviceFlag> workers_;
  std::vector<CPUMaskFlag> cpu_masks_;
  std::vector<int> num_threads_;
  bool allow_worksteal_ = false;
  int availability_check_interval_ms_ = 30000;
};

class BackendConfigBuilder {
  friend class RuntimeConfigBuilder;

  virtual absl::Status IsValid(
      ErrorReporter* error_reporter = DefaultErrorReporter()) = 0;

 public:
  virtual absl::StatusOr<std::shared_ptr<BackendConfig>> Build(
      ErrorReporter* error_reporter = DefaultErrorReporter()) = 0;
};

#ifdef BAND_TFLITE
class TfLiteBackendConfigBuilder : public BackendConfigBuilder {
  friend class RuntimeConfigBuilder;

 public:
  absl::StatusOr<std::shared_ptr<BackendConfig>> Build(
      ErrorReporter* error_reporter = DefaultErrorReporter()) override;

 private:
  absl::Status IsValid(
      ErrorReporter* error_reporter = DefaultErrorReporter()) override;
};
#endif  // BAND_TFLITE

#ifdef BAND_GRPC
class GrpcBackendConfigBuilder : public BackendConfigBuilder {
  friend class RuntimeConfigBuilder;

 public:
  GrpcBackendConfigBuilder& AddHost(std::string host) {
    host_ = host;
    return *this;
  }
  GrpcBackendConfigBuilder& AddPort(int port) {
    port_ = port;
    return *this;
  }
  absl::StatusOr<std::shared_ptr<BackendConfig>> Build(
      ErrorReporter* error_reporter = DefaultErrorReporter()) override;

 private:
  absl::Status IsValid(
      ErrorReporter* error_reporter = DefaultErrorReporter()) override;

  std::string host_ = "localhost";
  int port_ = 50051;
};
#endif  // BAND_GRPC

class ResourceMonitorConfigBuilder {
  friend class RuntimeConfigBuilder;

 public:
  ResourceMonitorConfigBuilder& AddResourceMonitorLogPath(
      std::string log_path) {
    log_path_ = log_path;
    return *this;
  }
  ResourceMonitorConfigBuilder& AddResourceMonitorDeviceFreqPath(
      DeviceFlag device, std::string device_freq_path) {
    device_freq_paths_.insert({device, device_freq_path});
    return *this;
  }
  ResourceMonitorConfigBuilder& AddResourceMonitorIntervalMs(
      int monitor_interval_ms) {
    monitor_interval_ms_ = monitor_interval_ms;
    return *this;
  }

  absl::StatusOr<ResourceMonitorConfig> Build();

 private:
  absl::Status IsValid();

  std::string log_path_ = "";
  std::map<DeviceFlag, std::string> device_freq_paths_;
  int monitor_interval_ms_ = 10;
};

// Delegate for ConfigBuilders
class RuntimeConfigBuilder {
 public:
  // Add ProfileConfig
  RuntimeConfigBuilder& AddOnline(bool online) {
    profile_config_builder_.AddOnline(online);
    return *this;
  }
  RuntimeConfigBuilder& AddNumWarmups(int num_warmups) {
    profile_config_builder_.AddNumWarmups(num_warmups);
    return *this;
  }
  RuntimeConfigBuilder& AddNumRuns(int num_runs) {
    profile_config_builder_.AddNumRuns(num_runs);
    return *this;
  }
  RuntimeConfigBuilder& AddCopyComputationRatio(
      std::vector<int> copy_computation_ratio) {
    profile_config_builder_.AddCopyComputationRatio(copy_computation_ratio);
    return *this;
  }

  RuntimeConfigBuilder& AddSmoothingFactor(float smoothing_factor) {
    profile_config_builder_.AddSmoothingFactor(smoothing_factor);
    return *this;
  }
  RuntimeConfigBuilder& AddProfileDataPath(std::string profile_log_path) {
    profile_config_builder_.AddProfileDataPath(profile_log_path);
    return *this;
  }

  // Add PlannerConfig
  RuntimeConfigBuilder& AddPlannerLogPath(std::string planner_log_path) {
    planner_config_builder_.AddLogPath(planner_log_path);
    return *this;
  }
  RuntimeConfigBuilder& AddScheduleWindowSize(int schedule_window_size) {
    planner_config_builder_.AddScheduleWindowSize(schedule_window_size);
    return *this;
  }
  RuntimeConfigBuilder& AddSchedulers(std::vector<SchedulerType> schedulers) {
    planner_config_builder_.AddSchedulers(schedulers);
    return *this;
  }
  RuntimeConfigBuilder& AddPlannerCPUMask(CPUMaskFlag cpu_masks) {
    planner_config_builder_.AddCPUMask(cpu_masks);
    return *this;
  }

  // Add WorkerConfig
  RuntimeConfigBuilder& AddWorkers(std::vector<DeviceFlag> workers) {
    worker_config_builder_.AddWorkers(workers);
    return *this;
  }
  RuntimeConfigBuilder& AddWorkerCPUMasks(std::vector<CPUMaskFlag> cpu_masks) {
    worker_config_builder_.AddCPUMasks(cpu_masks);
    return *this;
  }
  RuntimeConfigBuilder& AddWorkerNumThreads(std::vector<int> num_threads) {
    worker_config_builder_.AddNumThreads(num_threads);
    return *this;
  }
  RuntimeConfigBuilder& AddAllowWorkSteal(bool allow_worksteal) {
    worker_config_builder_.AddAllowWorkSteal(allow_worksteal);
    return *this;
  }
  RuntimeConfigBuilder& AddAvailabilityCheckIntervalMs(
      int32_t availability_check_interval_ms) {
    worker_config_builder_.AddAvailabilityCheckIntervalMs(
        availability_check_interval_ms);
    return *this;
  }

  // Add RuntimeConfig
  RuntimeConfigBuilder& AddResourceMonitorLogPath(std::string log_path) {
    resource_monitor_config_builder_.AddResourceMonitorLogPath(log_path);
    return *this;
  }
  RuntimeConfigBuilder& AddResourceMonitorDeviceFreqPath(
      DeviceFlag device, std::string device_freq_path) {
    resource_monitor_config_builder_.AddResourceMonitorDeviceFreqPath(
        device, device_freq_path);
    return *this;
  }
  RuntimeConfigBuilder& AddResourceMonitorIntervalMs(int monitor_interval_ms) {
    resource_monitor_config_builder_.AddResourceMonitorIntervalMs(
        monitor_interval_ms);
    return *this;
  }
  RuntimeConfigBuilder& AddMinimumSubgraphSize(int minimum_subgraph_size) {
    minimum_subgraph_size_ = minimum_subgraph_size;
    return *this;
  }
  RuntimeConfigBuilder& AddSubgraphPreparationType(
      SubgraphPreparationType subgraph_preparation_type) {
    subgraph_preparation_type_ = subgraph_preparation_type;
    return *this;
  }
  RuntimeConfigBuilder& AddCPUMask(CPUMaskFlag cpu_mask) {
    cpu_mask_ = cpu_mask;
    return *this;
  }

  // Add BackendConfig
#ifdef BAND_TFLITE
#endif  // BAND_TFLITE

#ifdef BAND_GRPC
  RuntimeConfigBuilder& AddGrpcHost(std::string host) {
    grpc_backend_config_builder_.AddHost(host);
    return *this;
  }
  RuntimeConfigBuilder& AddGrpcPort(int port) {
    grpc_backend_config_builder_.AddPort(port);
    return *this;
  }
#endif  // BAND_GRPC

  absl::StatusOr<RuntimeConfig> Build();

 private:
  absl::Status IsValid();

  ProfileConfigBuilder profile_config_builder_;
  PlannerConfigBuilder planner_config_builder_;
  WorkerConfigBuilder worker_config_builder_;
#ifdef BAND_TFLITE
  TfLiteBackendConfigBuilder tflite_backend_config_builder_;
#endif  // BAND_TFLITE
#ifdef BAND_GRPC
  GrpcBackendConfigBuilder grpc_backend_config_builder_;
#endif  // BAND_GRPC
  ResourceMonitorConfigBuilder resource_monitor_config_builder_;
  int minimum_subgraph_size_ = 7;
  SubgraphPreparationType subgraph_preparation_type_ =
      SubgraphPreparationType::kMergeUnitSubgraph;
  CPUMaskFlag cpu_mask_ = CPUMaskFlag::kAll;
};

}  // namespace band

#endif  // BAND_CONFIG_BUILDER_H_
