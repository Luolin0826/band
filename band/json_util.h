#ifndef BAND_JSON_UTIL_H_
#define BAND_JSON_UTIL_H_

#include <json/json.h>

#include <type_traits>
#include <typeinfo>

#include "band/common.h"
#include "band/logger.h"

#include "absl/status/status.h"

namespace band {
namespace json {
// load data from the given file
// if there is no such file, then the json object will be empty
// 文件存在性检查
Json::Value LoadFromFile(std::string file_path);
// write json object
// 写对象
absl::Status WriteToFile(const Json::Value& json_object, std::string file_path);
// validate the root, returns true if root is valid and has all required fields
// 验证根节点，如果根节点有效且具有所有必需字段，则返回true
bool Validate(const Json::Value& root, std::vector<std::string> required);
template <typename T>
bool AssignIfValid(T& lhs, const Json::Value& value, const char* key) {
  // 赋值验证
  if (!value[key].isNull()) {
    lhs = value[key].as<T>();
    return true;
  } else {
    return false;
  }
}
}  // namespace json
}  // namespace band

#endif