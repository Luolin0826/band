#ifndef TENSORFLOW_LITE_SHORTEST_EXPECTED_LATENCY_PLANNER_H_
#define TENSORFLOW_LITE_SHORTEST_EXPECTED_LATENCY_PLANNER_H_

#include "tensorflow/lite/planner.h"
#include "tensorflow/lite/interpreter.h"

namespace tflite {

namespace impl {

class Interpreter;

// 
class ShortestExpectedLatencyPlanner : public Planner {
 public:
  explicit ShortestExpectedLatencyPlanner(Interpreter* interpreter)
    : Planner(interpreter) {}
  void Plan() override;
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_SHORTEST_EXPECTED_LATENCY_PLANNER_H_
