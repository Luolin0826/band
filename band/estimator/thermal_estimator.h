#ifndef BAND_ESTIMATOR_THERMAL_ESTIMATOR_H_
#define BAND_ESTIMATOR_THERMAL_ESTIMATOR_H_

#include <chrono>
#include <deque>
#include <unordered_map>

#include "absl/status/status.h"
#include "band/common.h"
#include "band/config.h"
#include "band/estimator/estimator_interface.h"

#ifdef BAND_FREQ
#include "band/estimator/frequency_latency_estimator.h"
#else
#include "band/estimator/latency_estimator.h"
#endif  // BAND_FREQ

#include "band/profiler/frequency_profiler.h"
#include "band/profiler/latency_profiler.h"
#include "band/profiler/thermal_profiler.h"
#include "json/json.h"
#include "third_party/eigen3/Eigen/Cholesky"
#include "third_party/eigen3/Eigen/Core"

namespace band {

class ThermalEstimator
    : public IEstimator<SubgraphKey, ThermalInterval, ThermalMap> {
 public:
  explicit ThermalEstimator(
      IEngine* engine, ThermalProfiler* thermal_profiler,
      FrequencyProfiler* frequency_profiler, LatencyProfiler* latency_profiler,
#ifdef BAND_FREQ
      FrequencyLatencyEstimator* latency_estimator
#else
      LatencyEstimator* latency_estimator
#endif  // BAND_FREQ
      )
      : IEstimator(engine),
        thermal_profiler_(thermal_profiler),
        frequency_profiler_(frequency_profiler),
        latency_profiler_(latency_profiler),
        latency_estimator_(latency_estimator) {
  }
  absl::Status Init(const ThermalProfileConfig& config);
  void Update(const SubgraphKey& key, ThermalMap therm_start,
              ThermalMap therm_end, FreqMap freq, double latency);
  void UpdateWithEvent(const SubgraphKey& key, size_t event_handle) override;

  ThermalMap GetProfiled(const SubgraphKey& key) const override;
  ThermalMap GetExpected(const SubgraphKey& key) const override;

  absl::Status LoadModel(std::string profile_path) override;
  absl::Status DumpModel(std::string profile_path) override;

  Eigen::MatrixXd SolveLinear(Eigen::MatrixXd x, Eigen::MatrixXd y) {
    return (x.transpose() * x).ldlt().solve(x.transpose() * y);
  }

  Json::Value EigenMatrixToJson(Eigen::MatrixXd matrix);

  Eigen::MatrixXd JsonToEigenMatrix(Json::Value json);

 private:
  ThermalProfiler* thermal_profiler_;
  FrequencyProfiler* frequency_profiler_;
  LatencyProfiler* latency_profiler_;
#ifdef BAND_FREQ
  FrequencyLatencyEstimator* latency_estimator_;
#else
  LatencyEstimator* latency_estimator_;
#endif  // BAND_FREQ

  size_t num_resources_ = 0;
  size_t window_size_;

  Eigen::MatrixXd model_;
  std::deque<std::pair<Eigen::VectorXd, Eigen::VectorXd>> features_;
  mutable std::map<SubgraphKey, ThermalMap> profile_database_;
};

}  // namespace band

#endif  // BAND_ESTIMATOR_THERMAL_ESTIMATOR_H_