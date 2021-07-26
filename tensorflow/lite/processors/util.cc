#include "tensorflow/lite/processors/util.h"

#include <fstream>
#include <sstream>

namespace tflite {
namespace impl {
template <typename T>
T TryRead(std::vector<std::string> paths) {
  for (const std::string& path : paths) {
    std::fstream fs(path, std::fstream::in);
    if (fs.is_open()) {
      T output;
      fs >> output;
      return output;
    }
  }
  return T();
}

int TryReadInt(std::vector<std::string> paths) { return TryRead<int>(paths); }

std::vector<int> TryReadInts(std::vector<std::string> paths) {
  for (const std::string& path : paths) {
    std::fstream fs(path, std::fstream::in);
    if (fs.is_open()) {
      std::vector<int> outputs;
      int output;
      while (fs >> output) {
        outputs.push_back(output);
      }
      return outputs;
    }
  }
  return {};
}

std::string TryReadString(std::vector<std::string> paths) {
  return TryRead<std::string>(paths);
}
}  // namespace impl
}  // namespace tflite