#ifndef BAND_BACKEND_TFL_MODEL_EXECUTOR_H_
#define BAND_BACKEND_TFL_MODEL_EXECUTOR_H_

#include "band/interface/model_executor.h"
#include "tensorflow/lite/interpreter.h"

namespace Band {
namespace TfLite {
class TfLiteModelExecutor : public Interface::IModelExecutor {
 public:
  TfLiteModelExecutor(ModelId model_id, WorkerId worker_id,
                      DeviceFlags device_flag);
  ~TfLiteModelExecutor() override;

  absl::StatusOr<ModelSpec> InvestigateModelSpec(
      Interface::IModel* model) override;
  absl::Status PrepareSubgraph(Interface::IModel* model, std::set<int> ops = {},
                               std::set<int> unit_indices = {}) override;

  BackendType GetBackendType() const override;
  const std::vector<int>& GetInputs(const SubgraphKey& key) const override;
  const std::vector<int>& GetOutputs(const SubgraphKey& key) const override;
  const char* GetInputName(const SubgraphKey& key, int index) const override;
  const char* GetOutputName(const SubgraphKey& key, int index) const override;
  size_t GetNumTensors(const SubgraphKey& key) const override;
  size_t GetNumNodes(const SubgraphKey& key) const override;

  std::shared_ptr<Interface::ITensorView> GetTensorView(const SubgraphKey& key,
                                                        int index) override;
  SubgraphKey GetLargestSubgraphKey() const override;
  bool HasSubgraph(const SubgraphKey& key) const override;

  absl::Status ExecuteSubgraph(const SubgraphKey& key) override;
  void ForEachSubgraph(
      std::function<void(const SubgraphKey&)> iterator) override;

 private:
  friend class TfLiteUtil;

  tflite::Interpreter* GetInterpreter(const SubgraphKey& key);
  const tflite::Interpreter* GetInterpreter(const SubgraphKey& key) const;

  absl::StatusOr<std::unique_ptr<tflite::Interpreter>> CreateTfLiteInterpreter(
      Interface::IModel* model, DeviceFlags device,
      std::set<int> op_indices = {});
  static absl::StatusOr<TfLiteDelegate*> GetDeviceDelegate(DeviceFlags device);

  std::unordered_map<SubgraphKey, std::unique_ptr<tflite::Interpreter>,
                     SubgraphHash>
      interpreters_;
  static std::map<DeviceFlags, tflite::Interpreter::TfLiteDelegatePtr>
      delegates_;
};
}  // namespace TfLite
}  // namespace Band

#endif  // BAND_BACKEND_TFL_MODEL_EXECUTOR_H_
