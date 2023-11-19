#include "band/tool/benchmark.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>

#include "band/config_builder.h"
#include "band/logger.h"
#include "band/model.h"
#include "band/tensor.h"
#include "band/time.h"
#include "benchmark.h"

namespace band {
namespace tool {
Benchmark::Benchmark(BackendType target_backend)
    : target_backend_(target_backend) {}

Benchmark::~Benchmark() {
  if (runtime_config_) {
    delete runtime_config_;
  }
  for (auto model_context : model_contexts_) {
    delete model_context;
  }
}

absl::Status Benchmark::Run() {
  if (benchmark_config_.execution_mode == "periodic") {
    RunPeriodic();
  } else if (benchmark_config_.execution_mode == "stream") {
    RunStream();
  } else if (benchmark_config_.execution_mode == "workload") {
    RunWorkload();
  } else if (benchmark_config_.execution_mode == "all") {
    RunAll();
  } else if (benchmark_config_.execution_mode == "cpu") {
    RunCPU();
  } else if (benchmark_config_.execution_mode == "gpu") {
    RunGPU();
  } else if (benchmark_config_.execution_mode == "dsp") {
    RunDSP();
  } else if (benchmark_config_.execution_mode == "npu") {
    RunNPU();
  }

  return LogResults();
}

Benchmark::ModelContext::~ModelContext() {
  auto delete_tensors = [](Tensors& tensors) {
    for (auto t : tensors) {
      delete t;
    }
  };

  for (auto request_inputs : model_request_inputs) {
    delete_tensors(request_inputs);
  }

  for (auto request_outputs : model_request_outputs) {
    delete_tensors(request_outputs);
  }

  delete_tensors(model_inputs);
}

absl::Status Benchmark::ModelContext::PrepareInput() {
  for (int batch_index = 0; batch_index < model_request_inputs.size();
       batch_index++) {
    for (int input_index = 0; input_index < model_inputs.size();
         input_index++) {
      auto status =
          model_request_inputs[batch_index][input_index]->CopyDataFrom(
              model_inputs[input_index]);
      if (!status.ok()) {
        return status;
      }
    }
  }
  return absl::OkStatus();
}

bool Benchmark::ParseArgs(int argc, const char** argv) {
  if (argc < 2) {
    std::cout << "Usage:\n\tbenchmark <config-json-path> [<verbosity> = "
                 "default value: WARNING]"
              << std::endl;
    std::cout << "List of valid verbosity levels:" << std::endl;
    for (int i = 0; i < BAND_LOG_NUM_SEVERITIES; i++) {
      std::cout << "\t" << i << " : "
                << band::Logger::GetSeverityName(static_cast<LogSeverity>(i))
                << std::endl;
    }
    return false;
  }
  if (argc >= 3) {
    band::Logger::SetVerbosity(atoi(argv[2]));
  } else {
    band::Logger::SetVerbosity(BAND_LOG_INFO);
  }

  Json::Value json_config = json::LoadFromFile(argv[1]);
  return LoadBenchmarkConfigs(json_config) && LoadRuntimeConfigs(json_config);
}

bool Benchmark::LoadBenchmarkConfigs(const Json::Value& root) {
  if (!json::Validate(root, {"execution_mode", "models"})) {
    return false;
  }

  json::AssignIfValid(benchmark_config_.execution_mode, root, "execution_mode");

  std::set<std::string> supported_execution_modes{
      "periodic", "stream", "all", "cpu", "gpu", "dsp", "npu"};
  if (supported_execution_modes.find(benchmark_config_.execution_mode) ==
      supported_execution_modes.end()) {
    std::cout << "Please check if argument execution mode "
              << benchmark_config_.execution_mode << " is valid" << std::endl;
    return false;
  }

  json::AssignIfValid(benchmark_config_.running_time_ms, root,
                      "running_time_ms");

  if (benchmark_config_.running_time_ms == 0) {
    std::cout << "Please check if argument running_time_ms "
              << benchmark_config_.running_time_ms << " >= 0" << std::endl;
    return false;
  }

  if (root["models"].size() == 0) {
    std::cout << "Please specify at list one model in `models` argument"
              << std::endl;
    return false;
  }

  // Set Model Configurations
  for (int i = 0; i < root["models"].size(); ++i) {
    ModelConfig model;
    Json::Value model_json_value = root["models"][i];

    // Set model filepath.
    // Required for all cases.
    if (!json::AssignIfValid(model.path, model_json_value, "graph")) {
      std::cout
          << "Please check if argument `graph` is given in the model configs"
          << std::endl;
      return false;
    }

    // Set `period_ms`.
    // Required for `periodic` mode.
    if (benchmark_config_.execution_mode == "periodic") {
      if (!json::AssignIfValid(model.period_ms, model_json_value,
                               "period_ms") ||
          model.period_ms == 0) {
        std::cout << "Please check if argument `period_ms` is given and >= 0"
                  << std::endl;
        return false;
      }
    }

    json::AssignIfValid(model.batch_size, model_json_value, "batch_size");
    json::AssignIfValid(model.worker_id, model_json_value, "worker_id");
    json::AssignIfValid(model.slo_us, model_json_value, "slo_us");
    json::AssignIfValid(model.slo_scale, model_json_value, "slo_scale");
    json::AssignIfValid(model.runtime_frequency, model_json_value,
                        "runtime_frequency");
    json::AssignIfValid(model.cpu_frequency, model_json_value, "cpu_frequency");
    json::AssignIfValid(model.cpu_frequency, model_json_value, "gpu_frequency");

    benchmark_config_.model_configs.push_back(model);
  }
  return true;
}

bool tool::Benchmark::LoadRuntimeConfigs(const Json::Value& root) {
  if (!json::Validate(root, {"schedulers"})) {
    return false;
  }

  RuntimeConfigBuilder builder;
  // Profile config
  {
    if (root["profile_warmup_runs"].isNumeric()) {
      builder.AddNumWarmups(root["profile_warmup_runs"].asFloat());
    }
    if (root["profile_num_runs"].isInt()) {
      builder.AddNumRuns(root["profile_num_runs"].asInt());
    }
    if (root["profile_smoothing_factor"].isNumeric()) {
      builder.AddLatencySmoothingFactor(
          root["profile_smoothing_factor"].asFloat());
    }
    if (root["latency_profile_path"].isString()) {
      builder.AddProfilePath(root["latency_profile_path"].asCString());
    }
  }

  // Planner config
  {
    if (root["schedule_window_size"].isInt()) {
      builder.AddScheduleWindowSize(root["schedule_window_size"].asInt());
    }

    std::vector<SchedulerType> schedulers;
    for (auto scheduler : root["schedulers"]) {
      if (!scheduler.isString()) {
        BAND_LOG_PROD(BAND_LOG_ERROR,
                      "Please check if given scheduler is valid");
        return false;
      }
      schedulers.push_back(FromString<SchedulerType>(scheduler.asCString()));
    }
    builder.AddSchedulers(schedulers);

    if (root["cpu_masks"].isString()) {
      builder.AddCPUMask(
          FromString<CPUMaskFlag>(root["cpu_masks"].asCString()));
    }

    if (root["log_path"].isString()) {
      builder.AddPlannerLogPath(root["log_path"].asCString());
    }
  }

  // Worker config
  {
    if (!root["workers"].isNull()) {
      std::vector<DeviceFlag> workers;
      std::vector<CPUMaskFlag> cpu_masks;
      std::vector<int> num_threads;

      for (auto worker : root["workers"]) {
        if (worker["device"].isString()) {
          workers.push_back(
              FromString<DeviceFlag>(worker["device"].asCString()));
        }
        if (worker["num_threads"].isInt()) {
          num_threads.push_back(worker["num_threads"].asInt());
        }
        if (worker["cpu_masks"].isString()) {
          cpu_masks.push_back(
              FromString<CPUMaskFlag>(worker["cpu_masks"].asCString()));
        }
      }

      builder.AddWorkers(workers);
      builder.AddWorkerCPUMasks(cpu_masks);
      builder.AddWorkerNumThreads(num_threads);
    }

    if (root["availability_check_interval_ms"].isInt()) {
      builder.AddAvailabilityCheckIntervalMs(
          root["availability_check_interval_ms"].asInt());
    }
  }

  // Device config
  {
    if (root["cpu_therm_index"].isInt()) {
      builder.AddCPUThermIndex(root["cpu_therm_index"].asInt());
    }
    if (root["gpu_therm_index"].isInt()) {
      builder.AddGPUThermIndex(root["gpu_therm_index"].asInt());
    }
    if (root["dsp_therm_index"].isInt()) {
      builder.AddDSPThermIndex(root["dsp_therm_index"].asInt());
    }
    if (root["npu_therm_index"].isInt()) {
      builder.AddNPUThermIndex(root["npu_therm_index"].asInt());
    }
    if (root["target_therm_index"].isInt()) {
      builder.AddTargetThermIndex(root["target_therm_index"].asInt());
    }

    if (root["cpu_freq_path"].isString()) {
      builder.AddCPUFreqPath(root["cpu_freq_path"].asCString());
    }
    if (root["gpu_freq_path"].isString()) {
      builder.AddGPUFreqPath(root["gpu_freq_path"].asCString());
    }
    if (root["dsp_freq_path"].isString()) {
      builder.AddDSPFreqPath(root["dsp_freq_path"].asCString());
    }
    if (root["npu_freq_path"].isString()) {
      builder.AddNPUFreqPath(root["npu_freq_path"].asCString());
    }
    if (root["runtime_freq_path"].isString()) {
      builder.AddRuntimeFreqPath(root["runtime_freq_path"].asCString());
    }

    if (root["latency_log_path"].isString()) {
      builder.AddLatencyLogPath(root["latency_log_path"].asCString());
    }
    if (root["therm_log_path"].isString()) {
      builder.AddThermLogPath(root["therm_log_path"].asCString());
    }
    if (root["freq_log_path"].isString()) {
      builder.AddFreqLogPath(root["freq_log_path"].asCString());
    }
  }

  // ThermalProfile config
  {
    if (root["thermal_profile_window_size"].isInt()) {
      builder.AddThermalWindowSize(root["thermal_profile_window_size"].asInt());
    }
  }

  // Runtime config
  {
    if (root["minimum_subgraph_size"].isInt()) {
      builder.AddMinimumSubgraphSize(root["minimum_subgraph_size"].asInt());
    }

    if (root["subgraph_preparation_type"].isString()) {
      builder.AddSubgraphPreparationType(FromString<SubgraphPreparationType>(
          root["subgraph_preparation_type"].asCString()));
    }

    if (root["cpu_masks"].isString()) {
      builder.AddCPUMask(
          FromString<CPUMaskFlag>(root["cpu_masks"].asCString()));
    }
  }

  if (!builder.IsValid()) {
    std::cout << "Please check if given runtime config is valid" << std::endl;
    return false;
  }

  runtime_config_ = new RuntimeConfig(builder.Build());

  return true;
}

// motivated from /tensorflow/lite/tools/benchmark
template <typename T, typename Distribution>
void CreateRandomTensorData(void* target_ptr, int num_elements,
                            Distribution distribution) {
  std::mt19937 random_engine;
  T* target_head = static_cast<T*>(target_ptr);
  std::generate_n(target_head, num_elements, [&]() {
    return static_cast<T>(distribution(random_engine));
  });
}

absl::Status Benchmark::Initialize(int argc, const char** argv) {
  if (!ParseArgs(argc, argv)) {
    return absl::InternalError("Failed to parse arguments");
  }

  engine_ = Engine::Create(*runtime_config_);
  if (!engine_) {
    return absl::InternalError("Failed to create engine");
  }

  // load models
  for (auto& benchmark_model : benchmark_config_.model_configs) {
    ModelContext* engine = new ModelContext;

    {
      auto status =
          engine->model.FromPath(target_backend_, benchmark_model.path.c_str());
      if (!status.ok()) {
        return status;
      }
    }
    {
      auto status = engine_->RegisterModel(&engine->model);
      if (!status.ok()) {
        return status;
      }
    }

    const int model_id = engine->model.GetId();
    const auto input_indices = engine_->GetInputTensorIndices(model_id);
    const auto output_indices = engine_->GetOutputTensorIndices(model_id);

    for (int i = 0; i < benchmark_model.batch_size; i++) {
      // pre-allocate tensors
      Tensors inputs, outputs;
      for (int input_index : input_indices) {
        inputs.push_back(engine_->CreateTensor(model_id, input_index));
      }

      for (int output_index : output_indices) {
        outputs.push_back(engine_->CreateTensor(model_id, output_index));
      }

      engine->model_request_inputs.push_back(inputs);
      engine->model_request_outputs.push_back(outputs);
    }

    if (benchmark_model.slo_us <= 0 && benchmark_model.slo_scale > 0.f) {
      double worst_us = 0;

      // calculate worst case latency
      for (int worker_id = 0; worker_id < engine_->GetNumWorkers();
           worker_id++) {
        worst_us = std::max(engine_->GetProfiled(engine_->GetLargestSubgraphKey(
                                model_id, worker_id)),
                            worst_us);
      }

      if (worst_us == 0) {
        std::cout << "Failed to get worst case latency for model "
                  << benchmark_model.path << std::endl;
        std::cout << "Please check if given planner types require profiling"
                  << std::endl;
      } else {
        benchmark_model.slo_us = worst_us * benchmark_model.slo_scale;
      }
    }

    engine->model_ids =
        std::vector<ModelId>(benchmark_model.batch_size, model_id);
    engine->request_options = std::vector<RequestOption>(
        benchmark_model.batch_size, benchmark_model.GetRequestOption());

    // pre-allocate random input tensor to feed in run-time
    Tensors inputs;
    for (int input_index : input_indices) {
      interface::ITensor* input_tensor =
          engine_->CreateTensor(model_id, input_index);
      // random value ranges borrowed from tensorflow/lite/tools/benchmark
      switch (input_tensor->GetType()) {
        case DataType::kUInt8:
          CreateRandomTensorData<uint8_t>(
              input_tensor->GetData(), input_tensor->GetNumElements(),
              std::uniform_int_distribution<int32_t>(0, 254));
          break;
        case DataType::kInt8:
          CreateRandomTensorData<int8_t>(
              input_tensor->GetData(), input_tensor->GetNumElements(),
              std::uniform_int_distribution<int32_t>(-127, 127));
          break;
        case DataType::kInt16:
          CreateRandomTensorData<int16_t>(
              input_tensor->GetData(), input_tensor->GetNumElements(),
              std::uniform_int_distribution<int16_t>(0, 99));
          break;
        case DataType::kInt32:
          CreateRandomTensorData<int32_t>(
              input_tensor->GetData(), input_tensor->GetNumElements(),
              std::uniform_int_distribution<int32_t>(0, 99));
          break;
        case DataType::kInt64:
          CreateRandomTensorData<int64_t>(
              input_tensor->GetData(), input_tensor->GetNumElements(),
              std::uniform_int_distribution<int64_t>(0, 99));
          break;
        case DataType::kFloat32:
          CreateRandomTensorData<float>(
              input_tensor->GetData(), input_tensor->GetNumElements(),
              std::uniform_real_distribution<float>(-0.5, 0.5));
          break;
        case DataType::kFloat64:
          CreateRandomTensorData<double>(
              input_tensor->GetData(), input_tensor->GetNumElements(),
              std::uniform_real_distribution<double>(-0.5, 0.5));
          break;

        default:
          break;
      }
      inputs.push_back(input_tensor);
    }
    engine->model_inputs = inputs;
    model_contexts_.push_back(engine);
  }

  return absl::OkStatus();
}

void Benchmark::RunPeriodic() {
  std::thread t([this]() {
    const int64_t period_us =
        benchmark_config_.model_configs[0].period_ms * 1000;
    while (true) {
      auto start = std::chrono::system_clock::now();
      for (int model_index = 0; model_index < model_contexts_.size();
           model_index++) {
        ModelContext* model_context = model_contexts_[model_index];
        if (!model_context->PrepareInput().ok()) {
          BAND_LOG_PROD(BAND_LOG_WARNING, "Failed to prepare input");
          continue;
        }

        auto status = engine_->RequestSync(
            model_context->model_ids, model_context->request_options,
            model_context->model_request_inputs,
            model_context->model_request_outputs);
      }
      if (kill_app_) return;
      auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::system_clock::now() - start)
                            .count();
      if (elapsed_us < period_us) {
        std::this_thread::sleep_for(
            std::chrono::microseconds(period_us - elapsed_us));
      }
    }
  });

  t.detach();

  // wait for some time until we stop the benchmark
  std::this_thread::sleep_for(
      std::chrono::milliseconds(benchmark_config_.running_time_ms));
  kill_app_ = true;
  engine_->WaitAll();
}

void Benchmark::RunStream() {
  int run_duration_us = benchmark_config_.running_time_ms * 1000;
  int64_t start = time::NowMicros();
  while (true) {
    std::vector<ModelId> model_ids;
    std::vector<RequestOption> request_options;
    std::vector<Tensors> inputs;
    std::vector<Tensors> outputs;

    for (auto model_context : model_contexts_) {
      if (!model_context->PrepareInput().ok()) {
        BAND_LOG_PROD(BAND_LOG_WARNING, "Failed to prepare input");
        continue;
      }

      model_ids.insert(model_ids.end(), model_context->model_ids.begin(),
                       model_context->model_ids.end());
      request_options.insert(request_options.end(),
                             model_context->request_options.begin(),
                             model_context->request_options.end());
      inputs.insert(inputs.end(), model_context->model_request_inputs.begin(),
                    model_context->model_request_inputs.end());
      outputs.insert(outputs.end(),
                     model_context->model_request_outputs.begin(),
                     model_context->model_request_outputs.end());
    }

    auto status =
        engine_->RequestSync(model_ids, request_options, inputs, outputs);
    int64_t current = time::NowMicros();
    if (current - start >= run_duration_us) break;
  }
}

void Benchmark::RunWorkload() { BAND_NOT_IMPLEMENTED; }

void Benchmark::RunCPU() {
  for (int model_index = 0; model_index < model_contexts_.size();
       model_index++) {
    ModelContext* model_context = model_contexts_[model_index];
    for (auto cpu_freq : cpu_frequencies) {
      for (auto runtime_freq : runtime_frequencies) {
        for (auto& request_option : model_context->request_options) {
          request_option.target_worker = static_cast<int>(DeviceFlag::kCPU);
          request_option.runtime_frequency = runtime_freq;
          request_option.cpu_frequency = cpu_freq;
        }
        if (!model_context->PrepareInput().ok()) {
          BAND_LOG_PROD(BAND_LOG_WARNING, "Failed to prepare input");
          continue;
        }
        engine_->Sleep();
        auto status = engine_->RequestSync(
            model_context->model_ids, model_context->request_options,
            model_context->model_request_inputs,
            model_context->model_request_outputs);
      }
    }
  }

  engine_->WaitAll();
}

void Benchmark::RunGPU() {
  for (int model_index = 0; model_index < model_contexts_.size();
       model_index++) {
    ModelContext* model_context = model_contexts_[model_index];
    for (auto gpu_freq : gpu_frequencies) {
      for (auto runtime_freq : runtime_frequencies) {
        for (auto& request_option : model_context->request_options) {
          request_option.target_worker = static_cast<int>(DeviceFlag::kGPU);
          request_option.runtime_frequency = runtime_freq;
          request_option.gpu_frequency = gpu_freq;
        }
        if (!model_context->PrepareInput().ok()) {
          BAND_LOG_PROD(BAND_LOG_WARNING, "Failed to prepare input");
          continue;
        }
        engine_->Sleep();
        auto status = engine_->RequestSync(
            model_context->model_ids, model_context->request_options,
            model_context->model_request_inputs,
            model_context->model_request_outputs);
      }
    }
  }

  engine_->WaitAll();
}

void Benchmark::RunDSP() {
  for (int model_index = 0; model_index < model_contexts_.size();
       model_index++) {
    ModelContext* model_context = model_contexts_[model_index];
    for (auto runtime_freq : runtime_frequencies) {
      for (auto& request_option : model_context->request_options) {
        request_option.target_worker = static_cast<int>(DeviceFlag::kDSP);
        request_option.runtime_frequency = runtime_freq;
      }
      if (!model_context->PrepareInput().ok()) {
        BAND_LOG_PROD(BAND_LOG_WARNING, "Failed to prepare input");
        continue;
      }
      engine_->Sleep();
      auto status = engine_->RequestSync(model_context->model_ids,
                                         model_context->request_options,
                                         model_context->model_request_inputs,
                                         model_context->model_request_outputs);
    }
  }

  engine_->WaitAll();
}

void Benchmark::RunNPU() {
  for (int model_index = 0; model_index < model_contexts_.size();
       model_index++) {
    ModelContext* model_context = model_contexts_[model_index];
    for (auto runtime_freq : runtime_frequencies) {
      for (auto& request_option : model_context->request_options) {
        request_option.target_worker = static_cast<int>(DeviceFlag::kNPU);
        request_option.runtime_frequency = runtime_freq;
      }
      if (!model_context->PrepareInput().ok()) {
        BAND_LOG_PROD(BAND_LOG_WARNING, "Failed to prepare input");
        continue;
      }
      engine_->Sleep();
      auto status = engine_->RequestSync(model_context->model_ids,
                                         model_context->request_options,
                                         model_context->model_request_inputs,
                                         model_context->model_request_outputs);
    }
  }

  engine_->WaitAll();
}

void Benchmark::RunAll() {
  RunNPU();
  RunDSP();
  RunGPU();
  RunCPU();
}

void PrintHeader(std::string key, size_t indent_level = 0) {
  std::cout << std::left << std::string(indent_level * 2, ' ') << "<" << key
            << ">" << std::endl;
}

template <typename T>
void PrintLine(std::string key, const T& value, size_t indent_level = 0) {
  std::cout << std::left << std::string(indent_level * 2, ' ') << "[" << key
            << "] : " << std::right << value << std::endl;
}

absl::Status Benchmark::LogResults() {
  const std::string header = "--\t\t Band Benchmark Tool \t\t--";
  size_t length = header.size();
  std::cout << std::setfill('-') << std::setw(length) << std::fixed;
  std::cout << header << std::endl;

  PrintHeader("Option");
  PrintLine("Execution mode", benchmark_config_.execution_mode, 1);
  PrintLine("Running time (ms)", benchmark_config_.running_time_ms, 1);
  for (auto& scheduler : runtime_config_->planner_config.schedulers) {
    PrintLine("Scheduler", ToString(scheduler), 1);
  }

  PrintHeader("Model");
  for (auto& model_config : benchmark_config_.model_configs) {
    PrintHeader(model_config.path, 1);
    PrintLine("Batch size", model_config.batch_size, 2);
    PrintLine("Request period (ms)", model_config.period_ms, 2);
    PrintLine("SLO (us)", model_config.slo_us, 2);
    PrintLine("SLO scale", model_config.slo_scale, 2);
  }

  return absl::OkStatus();
}

}  // namespace tool
}  // namespace band