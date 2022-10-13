#ifndef BAND_PROFILER_H_
#define BAND_PROFILER_H_

#include <chrono>
#include <vector>

namespace Band {

class Profiler {
 public:
  size_t BeginEvent();
  void EndEvent(size_t event_handle);
  size_t GetNumEvents() const;

  template <typename T>
  uint64_t GetElapsedTimeAt(size_t index) {
    static_assert(is_chrono_duration<T>::value,
                  "T must be a std::chrono::duration");
    if (timeline_vector_.size() > index) {
      return std::chrono::duration_cast<T>(timeline_vector_[index].second -
                                           timeline_vector_[index].first)
          .count();
    } else
      return 0;
  }

  template <typename T>
  uint64_t GetAverageElapsedTime() {
    static_assert(is_chrono_duration<T>::value,
                  "T must be a std::chrono::duration");

    uint64_t accumulated_time = 0;
    for (size_t i = 0; i < timeline_vector_.size(); i++) {
      accumulated_time += GetElapsedTimeAt<T>(i);
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
}  // namespace Band
#endif