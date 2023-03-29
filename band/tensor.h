#ifndef BAND_TENSOR_H_
#define BAND_TENSOR_H_

#include <string>
#include <vector>

#include "band/interface/tensor.h"

namespace Band {
/*
  Tensor interface that tensor view / band tensor shares
*/

class Tensor : public Interface::ITensor {
 public:
  explicit Tensor(Interface::ITensor* tensor_view);
  ~Tensor();

  DataType GetType() const override;
  void SetType(DataType type) override;
  const char* GetData() const override;
  char* GetData() override;
  const int* GetDims() const override;
  size_t GetNumDims() const override;
  void SetDims(const std::vector<int>& dims) override;
  size_t GetBytes() const override;
  const char* GetName() const override;
  Quantization GetQuantization() const override;
  void SetQuantization(Quantization quantization) override;

 private:
  DataType type_;
  Quantization quantization_;
  size_t num_bytes_;
  std::vector<int> dims_;
  char* data_;
  std::string name_;
};
}  // namespace Band

#endif  // BAND_TENSOR_H_
