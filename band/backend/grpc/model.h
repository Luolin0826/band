#ifndef BAND_BACKEND_GRPC_MODEL_H_
#define BAND_BACKEND_GRPC_MODEL_H_

#include "band/interface/model.h"
#include "band/backend/grpc/proto/model.pb.h"

#include "absl/status/statusor.h"

namespace band {
namespace grpc {

class GrpcModel : public interface::IModel {
 public:
  GrpcModel(ModelId id);
  BackendType GetBackendType() const override;
  absl::Status FromPath(const char* filename) override;
  absl::Status FromBuffer(const char* buffer, size_t buffer_size) override;
  absl::Status FromProto(band_proto::ModelDescriptor proto);
  absl::Status ToPath(const char* filename) const;
  absl::StatusOr<band_proto::ModelDescriptor> ToProto() const;
  bool IsInitialized() const override;

  std::string id = "";
  int num_ops = -1;
  int num_tensors = -1;
  std::vector<DataType> tensor_types;
  std::vector<int> input_tensor_indices;
  std::vector<int> output_tensor_indices;
  std::vector<std::set<int>> op_input_tensors;
  std::vector<std::set<int>> op_output_tensors;
};

}  // namespace grpc
}  // namespace band

#endif  // BAND_BACKEND_GRPC_MODEL_H_