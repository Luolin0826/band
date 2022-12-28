#ifndef BAND_INTERFACE_INTERPRETER_H_
#define BAND_INTERFACE_INTERPRETER_H_

#include <memory>
#include <vector>

#include "band/common.h"
#include "band/interface/backend.h"
#include "band/interface/model.h"

namespace Band {
namespace Interface {
/*
  Interpreter for specific <IModel, processor>
*/

class ITensorView;

class IInterpreter : public IBackendSpecific {
 public:
  IInterpreter() = default;
  virtual ~IInterpreter() = default;
  // TODO: Subgraph generation
  virtual ModelSpec InvestigateModelSpec(IModel* model) = 0;
  virtual BandStatus FromModel(IModel* model, WorkerId worker_id,
                               BandDeviceFlags device,
                               std::set<int> ops = {}) = 0;

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
  virtual SubgraphKey GetLargestSubgraphKey(ModelId model_id) const = 0;

  virtual BandStatus InvokeSubgraph(const SubgraphKey& key) = 0;

 private:
  // Disable copy due to complexity
  IInterpreter(const IInterpreter&) = delete;
  IInterpreter(const IInterpreter&&) = delete;
  IInterpreter& operator=(const IInterpreter&) = delete;
  IInterpreter& operator=(const IInterpreter&&) = delete;
};
}  // namespace Interface
}  // namespace Band

#endif