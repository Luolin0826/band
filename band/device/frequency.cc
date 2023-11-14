#include "band/device/frequency.h"

#include <map>

#include "absl/strings/str_format.h"
#include "band/device/util.h"
#include "band/logger.h"

namespace band {

namespace {

std::string GetCpuFreqPath(const std::string& path) {
  return absl::StrFormat("%s/scaling_cur_freq", path.c_str());
}

std::string GetCpuScalingPath(const std::string& path) {
  return absl::StrFormat("%s/scaling_setspeed", path.c_str());
}

std::string GetCpuAvailableFreqPath(const std::string& path) {
  return absl::StrFormat("%s/scaling_available_frequencies", path.c_str());
}

std::string GetGpuFreqPath(const std::string& path) {
  return absl::StrFormat("%s/gpuclk", path.c_str());
}

std::string GetGpuMinScalingPath(const std::string& path) {
  return absl::StrFormat("%s/min_pwrlevel", path.c_str());
}

std::string GetGpuMaxScalingPath(const std::string& path) {
  return absl::StrFormat("%s/max_pwrlevel", path.c_str());
}

std::string GetGpuAvailableFreqPath(const std::string& path) {
  return absl::StrFormat("%s/devfreq/available_frequencies", path.c_str());
}

}  // anonymous namespace

Frequency::Frequency(DeviceConfig config) : config_(config) {
  device::Root();

  if (config.runtime_freq_path != "" &&
      CheckFrequency(config.runtime_freq_path)) {
    runtime_cpu_path_ = config.runtime_freq_path;
  } else {
    BAND_LOG_PROD(BAND_LOG_ERROR,
                  "Runtime frequency path \"%s\" is not available.",
                  config.cpu_freq_path.c_str());
  }

  if (config.cpu_freq_path != "" && CheckFrequency(config.cpu_freq_path)) {
    freq_device_map_[DeviceFlag::kCPU] = config.cpu_freq_path;
  } else {
    BAND_LOG_PROD(BAND_LOG_ERROR, "CPU frequency path \"%s\" is not available.",
                  config.cpu_freq_path.c_str());
  }

  if (config.gpu_freq_path != "" && CheckFrequency(config.gpu_freq_path)) {
    freq_device_map_[DeviceFlag::kGPU] = config.gpu_freq_path;
  } else {
    BAND_LOG_PROD(BAND_LOG_ERROR, "GPU frequency path \"%s\" is not available.",
                  config.gpu_freq_path.c_str());
  }
}

double Frequency::GetFrequency(DeviceFlag device_flag) {
  auto path = freq_device_map_[device_flag];
  if (device_flag == DeviceFlag::kCPU) {
    return device::TryReadDouble({GetCpuFreqPath(path)}, {cpu_freq_multiplier})
        .value();
  }
  return device::TryReadDouble({GetGpuFreqPath(path)}, {dev_freq_multiplier})
      .value();
}

double Frequency::GetRuntimeFrequency() {
  return device::TryReadDouble({GetCpuFreqPath(config_.runtime_freq_path)},
                               {cpu_freq_multiplier})
      .value();
}

absl::Status Frequency::SetRuntimeFrequency(double freq) {
  return SetFrequencyWithPath(GetCpuScalingPath(runtime_cpu_path_), freq,
                              cpu_freq_multiplier_w);
}

absl::Status Frequency::SetCpuFrequency(double freq) {
  return SetFrequencyWithPath(
      GetCpuScalingPath(freq_device_map_.at(DeviceFlag::kCPU)), freq,
      cpu_freq_multiplier_w);
}

absl::Status Frequency::SetGpuFrequency(double freq) {
  auto status1 = device::TryWriteSizeT(
      {GetGpuMinScalingPath(freq_device_map_.at(DeviceFlag::kGPU))},
      gpu_freq_map_.at(static_cast<size_t>(freq * dev_freq_multiplier_w)));
  auto status2 = device::TryWriteSizeT(
      {GetGpuMaxScalingPath(freq_device_map_.at(DeviceFlag::kGPU))},
      gpu_freq_map_.at(static_cast<size_t>(freq * dev_freq_multiplier_w)));
  auto status3 = device::TryWriteSizeT(
      {GetGpuMinScalingPath(freq_device_map_.at(DeviceFlag::kGPU))},
      gpu_freq_map_.at(static_cast<size_t>(freq * dev_freq_multiplier_w)));
  auto status4 = device::TryWriteSizeT(
      {GetGpuMaxScalingPath(freq_device_map_.at(DeviceFlag::kGPU))},
      gpu_freq_map_.at(static_cast<size_t>(freq * dev_freq_multiplier_w)));
  if (!status1.ok() || !status2.ok() || !status3.ok() || !status4.ok()) {
    return absl::InternalError("Failed to set GPU frequency.");
  }
  return absl::OkStatus();
}

absl::Status Frequency::SetFrequencyWithPath(const std::string& path,
                                             double freq, size_t multiplier) {
  return device::TryWriteSizeT({path}, static_cast<size_t>(freq * multiplier));
}

FreqMap Frequency::GetAllFrequency() {
  std::map<DeviceFlag, double> freq_map;
  for (auto& pair : freq_device_map_) {
    freq_map[pair.first] = GetFrequency(pair.first);
  }
  return freq_map;
}

std::map<DeviceFlag, std::vector<double>>
Frequency::GetAllAvailableFrequency() {
  if (freq_available_map_.size() > 0) {
    return freq_available_map_;
  }

  std::map<DeviceFlag, std::vector<double>> freq_map;
  for (auto& pair : freq_device_map_) {
    auto path = pair.second;
    if (pair.first == DeviceFlag::kCPU) {
      auto freqs = device::TryReadDoubles({GetCpuAvailableFreqPath(path)},
                                          {cpu_freq_multiplier})
                       .value();
      freq_map[pair.first] = freqs;
    } else {
      auto freqs = device::TryReadDoubles({GetGpuAvailableFreqPath(path)},
                                          {dev_freq_multiplier})
                       .value();
      freq_map[pair.first] = freqs;
    }
  }
  freq_available_map_ = freq_map;
  return freq_available_map_;
}

std::vector<double> Frequency::GetRuntimeAvailableFrequency() {
  if (freq_runtime_available_.size()) {
    return freq_runtime_available_;
  }

  freq_runtime_available_ =
      device::TryReadDoubles(
          {GetCpuAvailableFreqPath(config_.runtime_freq_path)},
          {cpu_freq_multiplier})
          .value();
  return freq_runtime_available_;
}

bool Frequency::CheckFrequency(std::string path) {
  return device::IsFileAvailable(path);
}

}  // namespace band