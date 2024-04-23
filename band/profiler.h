#ifndef BAND_PROFILER_H_
#define BAND_PROFILER_H_

#include <chrono>
#include <vector>

namespace band {

class Profiler {
 public:
  size_t BeginEvent();
  // 开始一个新的事件，记录事件开始时的时间点
  void EndEvent(size_t event_handle);
  // 结束一个已经开始的事件，记录事件结束时的时间点
  size_t GetNumEvents() const;
  // 获取已记录的事件数量

  template <typename T>
  double GetElapsedTimeAt(size_t index) const {
    // 获取特定事件的持续时间
    static_assert(is_chrono_duration<T>::value,
                  "T must be a std::chrono::duration");
    if (timeline_vector_.size() > index) {
      return std::max<double>(
          std::chrono::duration_cast<T>(timeline_vector_[index].second -
                                        timeline_vector_[index].first)
              .count(),
          0);
    } else
      return 0;
  }

  template <typename T>
  double GetAverageElapsedTime() const {
    // 计算所有事件的平均持续时间
    static_assert(is_chrono_duration<T>::value,
                  "T must be a std::chrono::duration");

    double accumulated_time = 0;
    for (size_t i = 0; i < timeline_vector_.size(); i++) {
      accumulated_time += GetElapsedTimeAt<T>(i);
    }

    if (timeline_vector_.size() == 0) {
      return 0;
    }

    return accumulated_time / timeline_vector_.size();
  }

 private:
  template <typename T>
  struct is_chrono_duration {
    static constexpr bool value = false;
  };

  template <typename Rep, typename Period>
  struct is_chrono_duration<std::chrono::duration<Rep, Period>> {
    static constexpr bool value = true;
  };

  std::vector<std::pair<std::chrono::system_clock::time_point,
                        std::chrono::system_clock::time_point>>
      timeline_vector_;
};
}  // namespace band
#endif