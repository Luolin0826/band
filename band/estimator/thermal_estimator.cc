#include "band/estimator/thermal_estimator.h"

#include "absl/strings/str_format.h"
#include "band/engine_interface.h"
#include "band/json_util.h"
#include "band/logger.h"
#include "band/model_spec.h"
#include "band/profiler.h"
#include "band/worker.h"

namespace band {

absl::Status ThermalEstimator::Init(const ThermalProfileConfig& config) {
  return absl::OkStatus();
}

void ThermalEstimator::Update(const ThermalSubgraphKey& key, int64_t latency) {
}

absl::Status ThermalEstimator::Profile(ModelId model_id) {
  return absl::OkStatus();
}

int64_t ThermalEstimator::GetProfiled(const ThermalSubgraphKey& key) const {
  return 0;
}

int64_t ThermalEstimator::GetExpected(const ThermalSubgraphKey& key) const {
  return 0;
}

absl::Status ThermalEstimator::DumpProfile() {
  return absl::OkStatus();
}

}  // namespace band