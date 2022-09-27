/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_JAVA_SRC_MAIN_NATIVE_JNI_UTILS_H_
#define TENSORFLOW_LITE_JAVA_SRC_MAIN_NATIVE_JNI_UTILS_H_

#include <jni.h>

#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/util.h"

extern const char kIllegalArgumentException[];
extern const char kIllegalStateException[];
extern const char kNullPointerException[];
extern const char kIndexOutOfBoundsException[];
extern const char kUnsupportedOperationException[];

namespace tflite {
namespace jni {
class TensorHandle {
 public:
  TensorHandle(TfLiteTensor* tensor);

  TfLiteTensor* tensor() const;

 private:
  TfLiteTensor* tensor_;
};

TfLiteTensor* GetTensorFromHandle(JNIEnv* env, jlong handle);

void ThrowException(JNIEnv* env, const char* clazz, const char* fmt, ...);

class BufferErrorReporter : public ErrorReporter {
 public:
  BufferErrorReporter(JNIEnv* env, int limit);
  virtual ~BufferErrorReporter();
  int Report(const char* format, va_list args) override;
  const char* CachedLastErrorMessage();

 private:
  char* buffer_;
  int limit_;
};

}  // namespace jni
}  // namespace tflite

#endif  // TENSORFLOW_LITE_JAVA_SRC_MAIN_NATIVE_JNI_UTILS_H_