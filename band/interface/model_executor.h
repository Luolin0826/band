#ifndef BAND_INTERFACE_MODEL_EXECUTOR_H_
#define BAND_INTERFACE_MODEL_EXECUTOR_H_

#include <functional>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "band/common.h"
#include "band/config.h"
#include "band/cpu.h"
#include "band/interface/backend.h"
#include "band/interface/model.h"
#include "band/model_spec.h"

namespace band {
namespace interface {
/*
  Model executor for specific <IModel, Worker>
*/

class ITensorView;
class IModelExecutor : public IBackendSpecific {
 public:
  IModelExecutor(
      ModelId model_id, WorkerId worker_id, DeviceFlags device_flag,
      const std::unique_ptr<BackendConfig>& backend_config,
      CpuSet thread_affinity_mask = BandCPUMaskGetSet(CPUMaskFlags::All),
      int num_threads = -1)
      : model_id_(model_id),
        worker_id_(worker_id),
        device_flag_(device_flag),
        backend_config_(backend_config),
        thread_affinity_mask_(thread_affinity_mask),
        num_threads_(num_threads > 0 ? num_threads : -1) {}
  virtual ~IModelExecutor() = default;

  virtual absl::StatusOr<ModelSpec> InvestigateModelSpec(IModel* model) = 0;
  virtual absl::Status PrepareSubgraph(IModel* model, std::set<int> ops = {},
                                       std::set<int> unit_indices = {}) = 0;

  virtual const std::vector<int>& GetInputs(const SubgraphKey& key) const = 0;
  virtual const std::vector<int>& GetOutputs(const SubgraphKey& key) const = 0;
  virtual const char* GetInputName(const SubgraphKey& key, int index) const = 0;
  virtual const char* GetOutputName(const SubgraphKey& key,
                                    int index) const = 0;
  virtual size_t GetNumTensors(const SubgraphKey& key) const = 0;
  virtual size_t GetNumNodes(const SubgraphKey& key) const = 0;

  virtual std::shared_ptr<ITensorView> GetTensorView(const SubgraphKey& key,
                                                     int index) = 0;

  virtual bool HasSubgraph(const SubgraphKey& key) const = 0;
  virtual SubgraphKey GetLargestSubgraphKey() const = 0;

  virtual absl::Status ExecuteSubgraph(const SubgraphKey& key) = 0;
  virtual void ForEachSubgraph(
      std::function<void(const SubgraphKey&)> iterator) = 0;

 protected:
  const ModelId model_id_;
  const WorkerId worker_id_;
  const DeviceFlags device_flag_;
  const std::unique_ptr<BackendConfig>& backend_config_;
  const CpuSet thread_affinity_mask_;
  const int num_threads_;

 private:
  // Disable copy due to complexity
  IModelExecutor(const IModelExecutor&) = delete;
  IModelExecutor(const IModelExecutor&&) = delete;
  IModelExecutor& operator=(const IModelExecutor&) = delete;
  IModelExecutor& operator=(const IModelExecutor&&) = delete;
};
}  // namespace interface
}  // namespace band

#endif