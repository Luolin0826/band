#include "band/tensor.h"

#include <string.h>

#include "band/logger.h"

namespace band {
Tensor::Tensor(ITensor* tensor_view, bool copy_data)
    : type_(tensor_view->GetType()),
      quantization_({QuantizationType::kNoQuantization, nullptr}),
      dims_(tensor_view->GetDims(),
            tensor_view->GetDims() + tensor_view->GetNumDims()),
      data_(new char[tensor_view->GetBytes()]),
      name_(tensor_view->GetName()) {
        // 从另一个张量对象 tensor_view 初始化一个 Tensor 对象。copy_data 参数指示是否进行数据的深拷贝
  auto status = SetQuantization(tensor_view->GetQuantization());
  if (!status.ok()) {
    BAND_LOG(LogSeverity::kError, "Failed to set quantization: %s",
                  status.message());
  }
  if (copy_data) {
    memcpy(data_, tensor_view->GetData(), tensor_view->GetBytes());
  }
}

Tensor::~Tensor() {
  delete[] data_;
  if (quantization_.GetParams() != nullptr) {
    free(quantization_.GetParams());
  }
}

DataType Tensor::GetType() const { return type_; }

void Tensor::SetType(DataType type) { type_ = type; }

const char* Tensor::GetData() const { return data_; }

char* Tensor::GetData() { return data_; }

const int* Tensor::GetDims() const { return dims_.data(); }

size_t Tensor::GetNumDims() const { return dims_.size(); }

void Tensor::SetDims(const std::vector<int>& dims) {
  dims_ = std::vector<int>(dims.begin(), dims.end());
}

const char* Tensor::GetName() const { return name_.c_str(); }

Quantization Tensor::GetQuantization() const { return quantization_; }

absl::Status Tensor::SetQuantization(Quantization quantization) {
  if (quantization_.GetType() == QuantizationType::kAffineQuantization) {
    // 检查是否需要释放之前的量化参数
    AffineQuantizationParams* input_q_params =
        reinterpret_cast<AffineQuantizationParams*>(quantization.GetParams());
        // 使用 reinterpret_cast 将 Quantization 对象的参数转换为 AffineQuantizationParams 结构体指针 input_q_params
        // 这允许直接访问仿射量化的参数

    AffineQuantizationParams* q_params =
        reinterpret_cast<AffineQuantizationParams*>(
            malloc(sizeof(AffineQuantizationParams)));
    if (input_q_params == nullptr || q_params == nullptr) {
      return absl::InternalError(
          "Failed to allocate memory for quantization params");
    }

    q_params->scale = std::vector<float>(input_q_params->scale.size());
    q_params->zero_point = std::vector<int>(input_q_params->zero_point.size());

    q_params->scale.insert(q_params->scale.end(), input_q_params->scale.begin(),
                           input_q_params->scale.end());
    q_params->zero_point.insert(q_params->zero_point.end(),
                                input_q_params->zero_point.begin(),
                                input_q_params->zero_point.end());
    q_params->quantized_dimension = input_q_params->quantized_dimension;
    quantization_.SetParams(q_params);
  }
  return absl::OkStatus();
}

}  // namespace band