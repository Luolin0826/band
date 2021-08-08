// This source code is created by Tencent's NCNN project.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#include "tensorflow/lite/processors/cpu.h"

#include <cstring>

#include "tensorflow/lite/processors/util.h"
#if defined __ANDROID__ || defined __linux__
#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace tflite {
namespace impl {

#if defined __ANDROID__ || defined __linux__
CpuSet::CpuSet() { DisableAll(); }

CpuSet CpuSet::GetCurrent() {
  CpuSet set;
  GetCPUThreadAffinity(set);
  return set;
}

void CpuSet::Enable(int cpu) { CPU_SET(cpu, &cpu_set_); }

void CpuSet::Disable(int cpu) { CPU_CLR(cpu, &cpu_set_); }

void CpuSet::DisableAll() { CPU_ZERO(&cpu_set_); }

bool CpuSet::IsEnabled(int cpu) const { return CPU_ISSET(cpu, &cpu_set_); }

int CpuSet::NumEnabled() const {
  int NumEnabled = 0;
  for (int i = 0; i < (int)sizeof(cpu_set_t) * 8; i++) {
    if (IsEnabled(i)) NumEnabled++;
  }

  return NumEnabled;
}
#else   // defined __ANDROID__ || defined __linux__
CpuSet::CpuSet() {}

CpuSet CpuSet::GetCurrent() { return {}; }

void CpuSet::Enable(int /* cpu */) {}

void CpuSet::Disable(int /* cpu */) {}

void CpuSet::DisableAll() {}

bool CpuSet::IsEnabled(int /* cpu */) const { return true; }

int CpuSet::NumEnabled() const { return GetCPUCount(); }
#endif  // defined __ANDROID__ || defined __linux__

static CpuSet g_thread_affinity_mask_all;
static CpuSet g_thread_affinity_mask_little;
static CpuSet g_thread_affinity_mask_big;
static CpuSet g_thread_affinity_mask_primary;
static int g_cpucount = GetCPUCount();

int GetCPUCount() {
  int count = 0;
#ifdef __EMSCRIPTEN__
  if (emscripten_has_threading_support())
    count = emscripten_num_logical_cores();
  else
    count = 1;
#elif defined __ANDROID__ || defined __linux__
  // get cpu count from /proc/cpuinfo
  FILE* fp = fopen("/proc/cpuinfo", "rb");
  if (!fp) return 1;

  char line[1024];
  while (!feof(fp)) {
    char* s = fgets(line, 1024, fp);
    if (!s) break;

    if (memcmp(line, "processor", 9) == 0) {
      count++;
    }
  }

  fclose(fp);
#elif __IOS__
  size_t len = sizeof(count);
  sysctlbyname("hw.ncpu", &count, &len, NULL, 0);
#else
  count = 1;
#endif

  if (count < 1) count = 1;

  return count;
}

int GetLittleCPUCount() {
  return TfLiteCPUMaskGetSet(kTfLiteLittle).NumEnabled();
}

int GetBigCPUCount() { return TfLiteCPUMaskGetSet(kTfLiteBig).NumEnabled(); }

namespace cpu {
int GetTargetMaxFrequencyKhz(int cpu) {
#if defined __ANDROID__ || defined __linux__
  return TryReadInt({"/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                         "/cpufreq/scaling_max_freq",
                     "/sys/devices/system/cpu/cpufreq/policy" +
                         std::to_string(cpu) + "/scaling_max_freq"});
#elif
  return -1;
#endif
}

int GetTargetMaxFrequencyKhz(const CpuSet& cpu_set) {
#if defined __ANDROID__ || defined __linux__
  if (cpu_set.NumEnabled() > 0) {
    int accumulated_frequency = 0;

    for (int i = 0; i < GetCPUCount(); i++) {
      if (cpu_set.IsEnabled(i))
        accumulated_frequency += GetTargetMaxFrequencyKhz(i);
    }

    return accumulated_frequency / cpu_set.NumEnabled();
  }
#endif
  return -1;
}

int GetTargetMinFrequencyKhz(int cpu) {
#if defined __ANDROID__ || defined __linux__
  return TryReadInt({"/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                         "/cpufreq/scaling_min_freq",
                     "/sys/devices/system/cpu/cpufreq/policy" +
                         std::to_string(cpu) + "/scaling_min_freq"});
#elif
  return -1;
#endif
}

int GetTargetMinFrequencyKhz(const CpuSet& cpu_set) {
#if defined __ANDROID__ || defined __linux__
  if (cpu_set.NumEnabled() > 0) {
    int accumulated_frequency = 0;

    for (int i = 0; i < GetCPUCount(); i++) {
      if (cpu_set.IsEnabled(i))
        accumulated_frequency += GetTargetMinFrequencyKhz(i);
    }

    return accumulated_frequency / cpu_set.NumEnabled();
  }
#endif
  return -1;
}

int GetTargetFrequencyKhz(int cpu) {
#if defined __ANDROID__ || defined __linux__
  return TryReadInt({"/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                         "/cpufreq/scaling_cur_freq",
                     "/sys/devices/system/cpu/cpufreq/policy" +
                         std::to_string(cpu) + "/scaling_cur_freq"});
#endif
  return -1;
}

int GetTargetFrequencyKhz(const CpuSet& cpu_set) {
#if defined __ANDROID__ || defined __linux__
  if (cpu_set.NumEnabled() > 0) {
    int accumulated_frequency = 0;

    for (int i = 0; i < GetCPUCount(); i++) {
      if (cpu_set.IsEnabled(i))
        accumulated_frequency += GetTargetFrequencyKhz(i);
    }

    return accumulated_frequency / cpu_set.NumEnabled();
  }
#endif
  return -1;
}

int GetFrequencyKhz(int cpu) {
#if defined __ANDROID__ || defined __linux__
  return TryReadInt({"/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                         "/cpufreq/cpuinfo_cur_freq",
                     "/sys/devices/system/cpu/cpufreq/policy" +
                         std::to_string(cpu) + "/cpuinfo_cur_freq"});
#elif
  return -1;
#endif
}

int GetFrequencyKhz(const CpuSet& cpu_set) {
#if defined __ANDROID__ || defined __linux__
  if (cpu_set.NumEnabled() > 0) {
    int accumulated_frequency = 0;

    for (int i = 0; i < GetCPUCount(); i++) {
      if (cpu_set.IsEnabled(i)) accumulated_frequency += GetFrequencyKhz(i);
    }

    return accumulated_frequency / cpu_set.NumEnabled();
  }
#elif
  return -1;
#endif
}

std::vector<int> GetAvailableFrequenciesKhz(const CpuSet& cpu_set) {
#if defined __ANDROID__ || defined __linux__
  for (int cpu = 0; cpu < GetCPUCount(); cpu++) {
    // Assuming that there is one cluster group
    if (cpu_set.IsEnabled(cpu)) {
      std::vector<int> frequencies = TryReadInts(
          {"/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
               "/cpufreq/scaling_available_frequencies",
           "/sys/devices/system/cpu/cpufreq/policy" + std::to_string(cpu) +
               "/scaling_available_frequencies"});
      if (frequencies.size()) {
        return frequencies;
      }
    }
  }

#endif
  return {};
}

int GetUpTransitionLatencyMs(int cpu) {
#if defined __ANDROID__ || defined __linux__
  int cpu_transition =
      TryReadInt({"/sys/devices/system/cpu/cpufreq/policy" +
                  std::to_string(cpu) + "/schedutil/up_rate_limit_us"}) /
      1000;
  if (cpu_transition == 0) {
    return TryReadInt({"/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                       "/cpufreq/cpuinfo_transition_latency"}) /
           1000000;
  } else {
    return cpu_transition;
  }
#endif
  return -1;
}

// Assuming that there is one cluster group
int GetUpTransitionLatencyMs(const CpuSet& cpu_set) {
#if defined __ANDROID__ || defined __linux__
  if (cpu_set.NumEnabled() > 0) {
    for (int i = 0; i < GetCPUCount(); i++) {
      if (cpu_set.IsEnabled(i)) {
        int transition_latency = GetUpTransitionLatencyMs(i);
        if (transition_latency > 0) {
          return transition_latency;
        }
      }
    }
  }
#endif
  return -1;
}

int GetDownTransitionLatencyMs(int cpu) {
#if defined __ANDROID__ || defined __linux__
  int cpu_transition =
      TryReadInt({"/sys/devices/system/cpu/cpufreq/policy" +
                  std::to_string(cpu) + "/schedutil/down_rate_limit_us"}) /
      1000;
  if (cpu_transition == 0) {
    return TryReadInt({"/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                       "/cpufreq/cpuinfo_transition_latency"}) /
           1000000;
  } else {
    return cpu_transition;
  }
#endif
  return -1;
}

// Assuming that there is one cluster group
int GetDownTransitionLatencyMs(const CpuSet& cpu_set) {
#if defined __ANDROID__ || defined __linux__
  for (int i = 0; i < GetCPUCount(); i++) {
    if (cpu_set.IsEnabled(i)) {
      int transition_latency = GetDownTransitionLatencyMs(i);
      if (transition_latency > 0) {
        return transition_latency;
      }
    }
  }
#endif
  return -1;
}

// Total transition count
// Note that cores in same cluster (little/big/primary)
// shares this value
int GetTotalTransitionCount(int cpu) {
#if defined __ANDROID__ || defined __linux__
  return TryReadInt({"/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                         "/cpufreq/stats/total_trans",
                     "/sys/devices/system/cpu/cpufreq/policy" +
                         std::to_string(cpu) + "/stats/total_trans"});
#endif
  return -1;
}

int GetTotalTransitionCount(const CpuSet& cpu_set) {
#if defined __ANDROID__ || defined __linux__
  if (cpu_set.NumEnabled() > 0) {
    int accumulated_transition_count = 0;

    for (int i = 0; i < GetCPUCount(); i++) {
      if (cpu_set.IsEnabled(i))
        accumulated_transition_count += GetTotalTransitionCount(i);
    }

    return accumulated_transition_count / cpu_set.NumEnabled();
  }
#elif
  return -1;
#endif
}

#if defined __ANDROID__ || defined __linux__
static int get_max_freq_khz(int cpuid) {
  // first try, for all possible cpu
  char path[256];
  sprintf(path, "/sys/devices/system/cpu/cpufreq/stats/cpu%d/time_in_state",
          cpuid);

  FILE* fp = fopen(path, "rb");

  if (!fp) {
    // second try, for online cpu
    sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/stats/time_in_state",
            cpuid);
    fp = fopen(path, "rb");

    if (fp) {
      int max_freq_khz = 0;
      while (!feof(fp)) {
        int freq_khz = 0;
        int nscan = fscanf(fp, "%d %*d", &freq_khz);
        if (nscan != 1) break;

        if (freq_khz > max_freq_khz) max_freq_khz = freq_khz;
      }

      fclose(fp);

      if (max_freq_khz != 0) return max_freq_khz;

      fp = NULL;
    }

    if (!fp) {
      // third try, for online cpu
      sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq",
              cpuid);
      fp = fopen(path, "rb");

      if (!fp) return -1;

      int max_freq_khz = -1;
      int nscan = fscanf(fp, "%d", &max_freq_khz);
      fclose(fp);

      return max_freq_khz;
    }
  }

  int max_freq_khz = 0;
  while (!feof(fp)) {
    int freq_khz = 0;
    int nscan = fscanf(fp, "%d %*d", &freq_khz);
    if (nscan != 1) break;

    if (freq_khz > max_freq_khz) max_freq_khz = freq_khz;
  }

  fclose(fp);

  return max_freq_khz;
}
}

int SetSchedAffinity(const CpuSet& thread_affinity_mask) {
  // set affinity for thread
#if defined(__GLIBC__) || defined(__OHOS__)
  pid_t pid = syscall(SYS_gettid);
#else
#ifdef PI3
  pid_t pid = getpid();
#else
  pid_t pid = gettid();
#endif
#endif

  int syscallret = syscall(__NR_sched_setaffinity, pid, sizeof(cpu_set_t),
                           &thread_affinity_mask.GetCpuSet());
  if (syscallret) {
    return -1;
  }

  return 0;
}

int GetSchedAffinity(CpuSet& thread_affinity_mask) {
  // set affinity for thread
#if defined(__GLIBC__) || defined(__OHOS__)
  pid_t pid = syscall(SYS_gettid);
#else
#ifdef PI3
  pid_t pid = getpid();
#else
  pid_t pid = gettid();
#endif
#endif

  int syscallret = syscall(__NR_sched_getaffinity, pid, sizeof(cpu_set_t),
                           &thread_affinity_mask.GetCpuSet());
  if (syscallret) {
    return -1;
  }

  return 0;
}
#endif  // defined __ANDROID__ || defined __linux__

TfLiteStatus SetCPUThreadAffinity(const CpuSet& thread_affinity_mask) {
#if defined __ANDROID__ || defined __linux__
  int num_threads = thread_affinity_mask.NumEnabled();
  int ssaret = SetSchedAffinity(thread_affinity_mask);
  if (ssaret != 0) return kTfLiteError;
#endif

  return kTfLiteOk;
}

TfLiteStatus GetCPUThreadAffinity(CpuSet& thread_affinity_mask) {
#if defined __ANDROID__ || defined __linux__
  int gsaret = GetSchedAffinity(thread_affinity_mask);
  if (gsaret != 0) return kTfLiteError;
#endif

  return kTfLiteOk;
}

int SetupThreadAffinityMasks() {
  g_thread_affinity_mask_all.DisableAll();

#if defined __ANDROID__ || defined __linux__
  int max_freq_khz_min = INT_MAX;
  int max_freq_khz_max = 0;
  std::vector<int> cpu_max_freq_khz(g_cpucount);
  for (int i = 0; i < g_cpucount; i++) {
    g_thread_affinity_mask_all.Enable(i);
    int max_freq_khz = cpu::get_max_freq_khz(i);

    cpu_max_freq_khz[i] = max_freq_khz;

    if (max_freq_khz > max_freq_khz_max) max_freq_khz_max = max_freq_khz;
    if (max_freq_khz < max_freq_khz_min) max_freq_khz_min = max_freq_khz;
  }

  int max_freq_khz_medium = (max_freq_khz_min + max_freq_khz_max) / 2;
  if (max_freq_khz_medium == max_freq_khz_max) {
    g_thread_affinity_mask_little.DisableAll();
    g_thread_affinity_mask_big = g_thread_affinity_mask_all;
    return 0;
  }

  for (int i = 0; i < g_cpucount; i++) {
    if (cpu_max_freq_khz[i] < max_freq_khz_medium) {
      g_thread_affinity_mask_little.Enable(i);
    } else if (cpu_max_freq_khz[i] == max_freq_khz_max) {
      g_thread_affinity_mask_primary.Enable(i);
    } else {
      g_thread_affinity_mask_big.Enable(i);
    }
  }

  // Categorize into LITTLE and big if there is no primary core.
  if (g_thread_affinity_mask_big.NumEnabled() == 0) {
    g_thread_affinity_mask_big = g_thread_affinity_mask_primary;
    g_thread_affinity_mask_primary.DisableAll();
  }

#else
  // TODO implement me for other platforms
  g_thread_affinity_mask_little.DisableAll();
  g_thread_affinity_mask_big = g_thread_affinity_mask_all;
#endif

  return 0;
}

const CpuSet& TfLiteCPUMaskGetSet(TfLiteCPUMaskFlags flag) {
  SetupThreadAffinityMasks();

  switch (flag) {
    case kTfLiteAll:
      return g_thread_affinity_mask_all;
    case kTfLiteLittle:
      return g_thread_affinity_mask_little;
    case kTfLiteBig:
      return g_thread_affinity_mask_big;
    case kTfLitePrimary:
      return g_thread_affinity_mask_primary;
    default:
      // fallback to all cores anyway
      return g_thread_affinity_mask_all;
  }
}

const char* TfLiteCPUMaskGetName(TfLiteCPUMaskFlags flag) {
  switch (flag) {
    case kTfLiteAll:
      return "ALL";
    case kTfLiteLittle:
      return "LITTLE";
    case kTfLiteBig:
      return "BIG";
    case kTfLitePrimary:
      return "PRIMARY";
    default:
      return "UNKNOWN";
  }
}

const TfLiteCPUMaskFlags TfLiteCPUMaskGetMask(const char* name) {
  for (int i = 0; i < kTfLiteNumCpuMasks; i++) {
    const auto flag = static_cast<TfLiteCPUMaskFlags>(i);
    if (strcmp(name, TfLiteCPUMaskGetName(flag)) == 0) {
      return flag;
    }
  }
  // Use all as a default flag
  return kTfLiteAll;
}

}  // namespace impl
}  // namespace tflite