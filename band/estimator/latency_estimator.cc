#include "band/estimator/latency_estimator.h"

#include "absl/strings/str_format.h"
#include "band/engine_interface.h"
#include "band/json_util.h"
#include "band/logger.h"
#include "band/model_spec.h"
#include "band/profiler/profiler.h"
#include "band/worker.h"

namespace band {
namespace {
int64_t ExpectWithMovingAverage(int64_t prev_latency, int64_t latency,
                                float smoothing_factor) {
  return smoothing_factor * latency + (1 - smoothing_factor) * prev_latency;
}
}  // anonymous namespace

LatencyEstimator::LatencyEstimator(IEngine* engine) : engine_(engine) {}

absl::Status LatencyEstimator::Init(const ProfileConfig& config) {
  profile_data_path_ = config.profile_data_path;
  if (!config.online) {
    profile_database_json_ = json::LoadFromFile(config.profile_data_path);
    if (profile_database_json_.empty()) {
      return absl::NotFoundError(absl::StrFormat(
          "Cannot find profile data file at %s. Please check the path.",
          config.profile_data_path.c_str()));
    }
  }
  // we cannot convert the model name strings to integer ids yet,
  // (profile_database_json_ --> profile_database_)
  // since we don't have anything in model_configs_ at the moment

  // Set how many runs are required to get the profile results.
  profile_online_ = config.online;
  profile_num_warmups_ = config.num_warmups;
  profile_num_runs_ = config.num_runs;
  profile_smoothing_factor_ = config.smoothing_factor;

  return absl::OkStatus();
}

absl::Status LatencyEstimator::UpdateLatency(const SubgraphKey& key,
                                             int64_t latency) {
  auto it = profile_database_.find(key);
  if (it == profile_database_.end()) {
    return absl::InternalError(absl::StrFormat(
        "The given SubgraphKey %s cannot be found.", key.ToString().c_str()));
  }
  int64_t prev_latency = it->second.expected;
  profile_database_[key].expected =
      ExpectWithMovingAverage(prev_latency, latency, profile_smoothing_factor_);
  return absl::OkStatus();
}

absl::Status LatencyEstimator::ProfileModel(ModelId model_id) {
  if (profile_online_) {
    for (WorkerId worker_id = 0; worker_id < engine_->GetNumWorkers();
         worker_id++) {
      Worker* worker = engine_->GetWorker(worker_id);
      // pause worker for profiling, must resume before continue
      worker->Pause();
      // wait for workers to finish current job
      worker->Wait();
      // invoke target subgraph in an isolated thread
      std::thread profile_thread([&]() {

#if BAND_IS_MOBILE
        if (worker->GetWorkerThreadAffinity().NumEnabled() > 0 &&
            !SetCPUThreadAffinity(worker->GetWorkerThreadAffinity()).ok()) {
          return absl::InternalError(absl::StrFormat(
              "Failed to propagate thread affinity of worker id "
              "%d to profile thread",
              worker_id));
        }
#endif

        engine_->ForEachSubgraph([&](const SubgraphKey& subgraph_key) -> void {
          if (subgraph_key.GetWorkerId() == worker_id &&
              subgraph_key.GetModelId() == model_id) {
            Profiler average_profiler;
            // TODO(#238): propagate affinity to CPU backend if necessary
            // (L1143-,tensorflow_band/lite/model_executor.cc)

            for (int i = 0; i < profile_num_warmups_; i++) {
              if (!engine_->Invoke(subgraph_key).ok()) {
                BAND_LOG_PROD(BAND_LOG_ERROR,
                              "Profiler failed to invoke largest subgraph of "
                              "model %d in worker %d",
                              model_id, worker_id);
              }
            }

            for (int i = 0; i < profile_num_runs_; i++) {
              const size_t event_id = average_profiler.BeginEvent();

              if (!engine_->Invoke(subgraph_key).ok()) {
                BAND_LOG_PROD(BAND_LOG_ERROR,
                              "Profiler failed to invoke largest subgraph of "
                              "model %d in worker %d",
                              model_id, worker_id);
              }
              average_profiler.EndEvent(event_id);
            }

            const int64_t latency =
                average_profiler
                    .GetAverageElapsedTime<std::chrono::microseconds>();

            BAND_LOG_PROD(BAND_LOG_INFO,
                          "Profiled latency of subgraph (%s) in worker "
                          "%d: %ld us",
                          subgraph_key.ToString().c_str(), worker_id, latency);

            profile_database_[subgraph_key] = {latency, latency};
          }
        });
        return absl::OkStatus();
      });

      profile_thread.join();

      // resume worker
      worker->Resume();
    }
  } else {
    if (engine_ && engine_->GetModelSpec(model_id)) {
      const std::string model_name = engine_->GetModelSpec(model_id)->path;
      auto model_profile = JsonToModelProfile(model_id);
      if (model_profile.size() > 0) {
        profile_database_.insert(model_profile.begin(), model_profile.end());
        BAND_LOG_PROD(
            BAND_LOG_INFO,
            "Successfully found %d profile entries for model (%s, %d).",
            model_profile.size(), model_name.c_str(), model_id);
      } else {
        BAND_LOG_PROD(BAND_LOG_WARNING,
                      "Failed to find profile entries for given model name %s.",
                      model_name.c_str());
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<LatencyRecord> LatencyEstimator::GetLatency(
    const SubgraphKey& key) const {
  auto it = profile_database_.find(key);
  if (it == profile_database_.end()) {
    return absl::InternalError(absl::StrFormat(
        "The given SubgraphKey %s cannot be found.", key.ToString().c_str()));
  }
  return it->second;
}

absl::StatusOr<int64_t> LatencyEstimator::GetProfiled(
    const SubgraphKey& key) const {
  auto status_or_latency = GetLatency(key);
  if (!status_or_latency.ok()) {
    return status_or_latency.status();
  }
  return status_or_latency.value().profiled;
}

absl::StatusOr<int64_t> LatencyEstimator::GetExpected(
    const SubgraphKey& key) const {
  auto status_or_latency = GetLatency(key);
  if (!status_or_latency.ok()) {
    return status_or_latency.status();
  }
  return status_or_latency.value().expected;
}

absl::StatusOr<int64_t> LatencyEstimator::GetWorst(ModelId model_id) const {
  int64_t worst_model_latency = 0;
  for (auto it : profile_database_) {
    if (it.first.GetModelId() == model_id) {
      worst_model_latency = std::max(worst_model_latency, it.second.expected);
    }
  }
  return worst_model_latency;
}

absl::Status LatencyEstimator::DumpProfile() {
  return json::WriteToFile(ProfileToJson(), profile_data_path_);
}

size_t LatencyEstimator::GetProfileHash() const {
  auto hash_func = std::hash<int>();
  std::size_t hash = hash_func(engine_->GetNumWorkers());
  for (int i = 0; i < engine_->GetNumWorkers(); i++) {
    hash ^= hash_func(static_cast<int>(engine_->GetWorker(i)->GetDeviceFlag()));
    hash ^= hash_func(engine_->GetWorker(i)->GetNumThreads());
    hash ^= hash_func(static_cast<int>(
        engine_->GetWorker(i)->GetWorkerThreadAffinity().GetCPUMaskFlag()));
  }
  return hash;
}

std::map<SubgraphKey, LatencyRecord>
LatencyEstimator::JsonToModelProfile(const int model_id) {
  auto string_to_indices = [](std::string index_string) {
    std::set<int> node_indices;
    std::stringstream ss(index_string);

    for (int i; ss >> i;) {
      node_indices.insert(i);
      if (ss.peek() == ',') {
        ss.ignore();
      }
    }

    return node_indices;
  };

  std::map<SubgraphKey, LatencyRecord> id_profile;
  if (profile_database_json_["hash"].asUInt64() != GetProfileHash()) {
    BAND_LOG_INTERNAL(
        BAND_LOG_WARNING,
        "Current profile hash does not matches with a file (%s). Will ignore.",
        profile_data_path_.c_str());
    return id_profile;
  }

  for (auto profile_it = profile_database_json_.begin();
       profile_it != profile_database_json_.end(); ++profile_it) {
    std::string model_name = profile_it.key().asString();

    const Json::Value idx_profile = *profile_it;
    for (auto idx_profile_it = idx_profile.begin();
         idx_profile_it != idx_profile.end(); ++idx_profile_it) {
      std::string unit_indices_string = idx_profile_it.key().asString();
      std::set<int> unit_indices = string_to_indices(unit_indices_string);

      const Json::Value device_profile = *idx_profile_it;
      for (auto device_profile_it = device_profile.begin();
           device_profile_it != device_profile.end(); ++device_profile_it) {
        int worker_id = device_profile_it.key().asInt();
        int64_t profiled_latency = (*device_profile_it).asInt64();

        if (profiled_latency <= 0) {
          // jsoncpp treats missing values (null) as zero,
          // so they will be filtered out here
          continue;
        }

        SubgraphKey key(model_id, worker_id, unit_indices);
        id_profile[key] = {profiled_latency, profiled_latency};
      }
    }
  }
  return id_profile;
}

Json::Value LatencyEstimator::ProfileToJson() {
  Json::Value name_profile;
  name_profile["hash"] = GetProfileHash();
  for (auto& pair : profile_database_) {
    SubgraphKey key = pair.first;
    const int model_id = key.GetModelId();
    const int64_t profiled_latency = pair.second.profiled;

    // check the string name of this model id
    auto model_spec = engine_->GetModelSpec(model_id);
    if (model_spec && !model_spec->path.empty()) {
      // copy all entries in id_profile --> database_json
      name_profile[model_spec->path][key.GetUnitIndicesString()]
                  [key.GetWorkerId()] = profiled_latency;
    } else {
      BAND_LOG_INTERNAL(BAND_LOG_ERROR,
                        "Cannot find model %d from "
                        "model_configs. Will ignore.",
                        model_id);
      continue;
    }
  }
  return name_profile;
}

}  // namespace band