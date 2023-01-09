#include <jni.h>

#include "band/config.h"
#include "band/java/src/main/native/jni_utils.h"

using Band::RuntimeConfig;
using Band::jni::JNIRuntimeConfig;

JNIEXPORT void JNICALL
Java_org_mrsnu_band_Config_deleteConfig(
    JNIEnv* env, jclass clazz, jlong configHandle) {
  delete reinterpret_cast<JNIRuntimeConfig*>(configHandle);
}