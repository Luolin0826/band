#ifndef BAND_JAVA_SRC_MAIN_NATIVE_JNI_UTILS_H_
#define BAND_JAVA_SRC_MAIN_NATIVE_JNI_UTILS_H_

#include <jni.h>

#include "band/config.h"
#include "band/config_builder.h"
#include "band/engine.h"
#include "band/model.h"
#include "band/tensor.h"
#include "band/error_reporter.h"

namespace Band {
namespace jni {

#define JNI_DEFINE_CLS(tag, cls)                                            \
  static jclass tag##_cls = env->FindClass(cls);                            \
  if (tag##_cls == nullptr) {                                               \
    BAND_LOG_INTERNAL(Band::BAND_LOG_ERROR, "Canont find class named `%s`", \
                      cls);                                                 \
  }

#define JNI_DEFINE_MTD(tag, cls_var, mtd, sig)                             \
  static jmethodID tag##_mtd = env->GetMethodID(cls_var, mtd, sig);        \
  if (tag##_mtd == nullptr) {                                              \
    BAND_LOG_INTERNAL(Band::BAND_LOG_ERROR,                                \
                      "Cannot find method named `%s` with signature `%s`", \
                      mtd, sig);                                           \
  }

#define JNI_DEFINE_CLS_AND_MTD(tag, cls, mtd, sig) \
  JNI_DEFINE_CLS(tag, cls)                         \
  JNI_DEFINE_MTD(tag, tag##_cls, mtd, sig);

extern const char kIllegalArgumentException[];
extern const char kNullPointerException[];

struct JNIRuntimeConfig {
  JNIRuntimeConfig(RuntimeConfig config) : impl(config) {}

  RuntimeConfig impl;
};

void ThrowException(JNIEnv* env, const char* clazz, const char* fmt, ...);

bool CheckJniInitializedOrThrow(JNIEnv* env);

class BufferErrorReporter : public ErrorReporter {
 public:
  BufferErrorReporter(JNIEnv* env, int limit);
  ~BufferErrorReporter() override;
  int Report(const char* format, va_list args);
  const char* CachedErrorMessage();
  using ErrorReporter::Report;

 private:
  char* buffer_;
  int start_idx_ = 0;
  int end_idx_ = 0;
};

template <typename T>
T* CastLongToPointer(JNIEnv* env, jlong handle) {
  if (handle == 0 || handle == -1) {
    ThrowException(env, Band::jni::kIllegalArgumentException,
                   "Internal error: Found invalid handle");
    return nullptr;
  }
  return reinterpret_cast<T*>(handle);
}

std::string ConvertJstringToString(JNIEnv* env, jstring jstr);

Engine* ConvertLongToEngine(JNIEnv* env, jlong handle);

RuntimeConfigBuilder* ConvertLongToConfigBuilder(JNIEnv* env, jlong handle);

RuntimeConfig* ConvertLongToConfig(JNIEnv* env, jlong handle);

Model* ConvertLongToModel(JNIEnv* env, jlong handle);

Tensor* ConvertLongToTensor(JNIEnv* env, jlong handle);

int ConvertLongToJobId(jint request_handle);

Tensors ConvertLongListToTensors(JNIEnv* env, jobject tensor_handles);

BufferErrorReporter* ConvertLongToErrorReporter(JNIEnv* env, jlong handle);

}  // namespace jni
}  // namespace Band

#endif  // BAND_JAVA_SRC_MAIN_NATIVE_JNI_UTILS_H_