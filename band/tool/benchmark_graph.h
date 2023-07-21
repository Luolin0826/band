#ifndef BAND_TOOL_BENCHMARK_GRAPH_H_
#define BAND_TOOL_BENCHMARK_GRAPH_H_

#include <map>
#include <mutex>
#include <string>

#include "absl/status/statusor.h"
#include "band/config.h"
#include "band/engine.h"
#include "band/json_util.h"

namespace band {
namespace tool {
class EngineRunner;

class BenchmarkGraph {
 public:
  BenchmarkGraph() = default;
  ~BenchmarkGraph();

  struct Vertex {
    Vertex(const Model& model, size_t batch_size, int worker_id,
           size_t vertex_id)
        : model(model),
          batch_size(batch_size),
          worker_id(worker_id),
          vertex_id(vertex_id) {}
    ~Vertex();

    void InitializeContext(Engine& engine);
    const RequestOption GetRequestOption() const;
    absl::Status PrepareInput();

    const Model& model;
    size_t batch_size;
    int worker_id;
    size_t vertex_id;

    // pre-allocated model tensors for runtime requests
    std::vector<ModelId> model_ids;
    std::vector<RequestOption> request_options;
    std::vector<Tensors> model_request_inputs;
    std::vector<Tensors> model_request_outputs;
    // randomly generated input
    Tensors model_inputs;
  };

  absl::Status Initialize(const Json::Value& root,
                          const EngineRunner& engine_runner);

  void InitializeExecutionContext();
  std::vector<const Vertex*> GetNextVertices() const;
  void OnVertexFinished(size_t vertex_id);
  bool IsFinished() const;

 private:
  std::set<size_t> GetResolvedVertexIds() const;
  absl::Status CheckCycles() const;

  std::vector<std::string> vertex_names_;
  std::vector<Vertex*> vertices_;
  std::vector<std::pair<size_t, size_t>> edges_;

  mutable std::mutex mutex_;
  std::set<size_t> finished_vertices_;
};

}  // namespace tool
}  // namespace band

#endif