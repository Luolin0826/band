#ifndef BAND_ESTIMATOR_ESTIMATOR_H_
#define BAND_ESTIMATOR_ESTIMATOR_H_

#include <chrono>
#include <unordered_map>

#include "band/common.h"
#include "band/config.h"

#include "absl/status/status.h"

namespace band {

class IEngine;

class IEstimator {
 public:
  explicit IEstimator(IEngine* engine) : engine_(engine) {}
  virtual absl::Status Init(const ProfileConfig& config) = 0;
  virtual void Update(const SubgraphKey& key, int64_t new_value) = 0;
  virtual absl::Status Profile(ModelId model_id) = 0;
  virtual int64_t GetProfiled(const SubgraphKey& key) const = 0;
  virtual int64_t GetExpected(const SubgraphKey& key) const = 0;
  virtual int64_t GetWorst(ModelId model_id) const = 0;

  virtual absl::Status DumpProfile() = 0;

 protected:
  IEngine* engine_;
};

}  // namespace band

#endif  // BAND_ESTIMATOR_ESTIMATOR_H_