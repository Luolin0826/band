#ifndef BAND_LATENCY_ESTIMATOR_H_
#define BAND_LATENCY_ESTIMATOR_H_

#include <json/json.h>

#include <chrono>
#include <unordered_map>

#include "absl/status/status.h"
#include "band/common.h"
#include "band/config.h"

namespace band {
class IEngine;
class LatencyEstimator {
  // 旨在测量、记录和预测计算模型中不同子图的执行时间（延迟）
 public:
  explicit LatencyEstimator(IEngine* engine);
  // 初始化 LatencyEstimator 实例，将引擎接口的指针存储为成员变量，以便在其他方法中使用
  absl::Status Init(const ProfileConfig& config);
  // 设置性能分析的配置，包括是否在线分析、预热次数、运行次数和平滑因子。
  // 加载指定路径的延迟分析数据（如果不是在线模式）。
  void UpdateLatency(const SubgraphKey& key, int64_t latency);
  // 更新指定子图的延迟数据。
  // 如果子图已存在于延迟数据库中，使用指定的平滑因子更新其移动平均延迟；如果不存在，则记录警告

  absl::Status ProfileModel(ModelId model_id);
  // 对指定模型进行性能分析。
  // 如果是在线模式，则挂起工作线程，单独运行子图并记录延迟；
  // 如果是离线模式，则从JSON数据中加载模型的延迟配置。
  int64_t GetProfiled(const SubgraphKey& key) const;
  // 返回给定子图的最近一次测量延迟
  int64_t GetExpected(const SubgraphKey& key) const;
  // 返回给定子图的预期（移动平均）延迟
  int64_t GetWorst(ModelId model_id) const;
  // 返回指定模型所有子图中最糟糕的延迟

  absl::Status DumpProfile();
  // 将当前的延迟分析数据写入到预设的文件路径

  // latency in microseconds
  // 在微秒中测量延迟
  struct Latency {
    int64_t profiled;
    int64_t moving_averaged;
  };

 private:
  size_t GetProfileHash() const;
  // 计算当前配置的哈希值，用于验证延迟数据的一致性

  // Convert entries in the json value to ModelDeviceToLatency format,
  // for the given model name and target model id.
  // 将JSON格式的数据转换为模型的延迟配置映射，用于从存储的数据中恢复延迟信息
  std::map<SubgraphKey, Latency> JsonToModelProfile(
      const std::string& model_fname, const int model_id);

  // Convert model integer ids back to string-type names for model profiles,
  // and returns the json format identical to `profile_database_json_`.
  // 将模型整数id转换回模型配置文件的字符串类型名称，并返回与`profile_database_json_`相同的json格式。
  Json::Value ProfileToJson();

  // Path to the profile data.
  // The data in the path will be read during initial phase, and also
  // will be updated at the end of the run.
  // 档案数据的路径。
  // 在初始阶段将读取路径中的数据，并且还将在运行结束时更新。
  std::string profile_data_path_;

  // The contents of the file at `profile_data_path_`.
  // We keep this separately from `profile_database_`, since we cannot
  // immediately put `profile_data_path_`'s contents into `profile_database_`
  // because the model name --> int mapping is not available at init time.
  // `profile_data_path_`处文件的内容。
  // 我们将其与`profile_database_`分开，因为我们不能立即将`profile_data_path_`的内容放入`profile_database_`中，
  // 因为在初始化时没有可用的模型名称-->整数映射。

  Json::Value profile_database_json_;

  std::unordered_map<SubgraphKey, Latency, SubgraphHash> profile_database_;
  float profile_smoothing_factor_ = 0.05f;

  bool profile_online_;
  int profile_num_warmups_;
  int profile_num_runs_;

  IEngine* const engine_;
};
}  // namespace band

#endif  // BAND_LATENCY_ESTIMATOR_H_