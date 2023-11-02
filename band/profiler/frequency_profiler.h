#ifndef BAND_PROFILER_FREQUECY_PROFILER_H_
#define BAND_PROFILER_FREQUECY_PROFILER_H_

#include <chrono>
#include <fstream>
#include <vector>

#include "band/config.h"
#include "band/device/frequency.h"
#include "band/profiler/profiler.h"
#include "band/logger.h"

namespace band {

using FreqInfo = std::pair<std::chrono::system_clock::time_point, FreqMap>;
using FreqInterval = std::pair<FreqInfo, FreqInfo>;

class FrequencyProfiler : public Profiler {
 public:
  explicit FrequencyProfiler(DeviceConfig config);
  ~FrequencyProfiler() {}

  size_t BeginEvent() override;
  void EndEvent(size_t event_handle) override;
  size_t GetNumEvents() const override;

  FreqInterval GetInterval(size_t index) const {
    if (timeline_.size() <= index) {
      return {};
    }
    return timeline_[index];
  }

  FreqInfo GetStart(size_t index) const { return GetInterval(index).first; }

  FreqInfo GetEnd(size_t index) const { return GetInterval(index).second; }

  FreqMap GetAllFrequency() const { return frequency_->GetAllFrequency(); }

  std::map<DeviceFlag, std::vector<double>> GetAllAvailableFrequency() const {
    return frequency_->GetAllAvailableFrequency();
  }

 private:
  std::unique_ptr<Frequency> frequency_;
  std::vector<std::pair<FreqInfo, FreqInfo>> timeline_;
};

}  // namespace band

#endif  // BAND_PROFILER_FREQUECY_PROFILER_H_