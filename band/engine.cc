#include "band/engine.h"

#include <algorithm>

#include "band/backend_factory.h"
#include "band/context.h"
#include "band/interface/tensor_view.h"
#include "band/latency_estimator.h"
#include "band/logger.h"
#include "band/model.h"
#include "band/model_analyzer.h"
#include "band/planner.h"
#include "band/tensor.h"
#include "band/worker.h"
#include "engine.h"

namespace Band {
Engine::~Engine() {
  for (auto& interpreter : interpreters_) {
    interpreter.second.reset();
  }

  for (auto& worker : workers_) {
    worker->End();
  }

  for (auto& worker : workers_) {
    worker.reset();
  }

  planner_.reset();
}

std::unique_ptr<Engine> Engine::Create(const RuntimeConfig& config,
                                       ErrorReporter* error_reporter) {
  std::unique_ptr<Engine> engine_ptr(new Engine(error_reporter));
  return engine_ptr->Init(config) == kBandOk ? std::move(engine_ptr) : nullptr;
}

BandStatus Engine::RegisterModel(Model* model) {
  if (!model || model->GetSupportedBackends().size() == 0) {
    BAND_LOG_INTERNAL(BAND_LOG_ERROR, "Failed to register model");
    return kBandError;
  }

  const ModelId model_id = model->GetId();

  for (BandBackendType backend_type : model->GetSupportedBackends()) {
    // Analyze model & generate subgraphs per backend type
    ModelAnalyzer analyzer(*this, planner_->NeedFallbackSubgraphs(),
                           model_config_, model, backend_type);
    const auto result = analyzer.CreateSubgraphs();
    // TODO: replace with a std::tie?
    const BandStatus status = std::get<0>(result);
    const ModelSpec model_spec = std::get<1>(result);
    const std::vector<SubgraphDef> subgraph_defs = std::get<2>(result);

    if (status != kBandOk) {
      BAND_REPORT_ERROR(
          error_reporter_,
          "Failed to create subgraphs for model %d with subgraph option %s",
          model_id,
          BandSubgraphPreparationGetName(
              model_config_.subgraph_preparation_type));
      // TODO(BAND-49): unregister for specific backend
      UnregisterModel(model);
      return kBandError;
    }

    BAND_LOG_PROD(
        BAND_LOG_INFO, "Create %d subgraphs for model %s with mode %s %s",
        subgraph_defs.size(), model_spec.path.c_str(),
        BandSubgraphPreparationGetName(model_config_.subgraph_preparation_type),
        SummarizeSubgraphs(subgraph_defs).c_str());

    // Create internal interpreter per each supported backends
    {
      bool added_once = false;
      for (WorkerId worker_id = 0; worker_id < workers_.size(); worker_id++) {
        if (model_spec.unavailable_devices.find(GetWorkerDevice(worker_id)) ==
            model_spec.unavailable_devices.end()) {
          std::unique_ptr<Interface::IInterpreter> interpreter(
              BackendFactory::CreateInterpreter(backend_type, model_id,
                                                worker_id,
                                                GetWorkerDevice(worker_id)));
          interpreters_[{model_id, worker_id}] = std::move(interpreter);
          added_once = true;
          BAND_LOG_INTERNAL(
              BAND_LOG_INFO, "Create interpreter for model %d worker %s (%s)",
              model_id, BandDeviceGetName(GetWorkerDevice(worker_id)),
              BandStatusGetName(status));
        }
      }

      if (!added_once) {
        BAND_REPORT_ERROR(error_reporter_,
                          "Failed to create interpreter on all worker types");
        // TODO(BAND-49): unregister for specific backend
        UnregisterModel(model);
        return kBandError;
      }
    }

    model_specs_.insert({model_id, model_spec});

    // Prepare execution of subgraph definitions per each interpreter
    {
      for (const SubgraphDef& subgraph_def : subgraph_defs) {
        const std::pair<ModelId, WorkerId> interpreter_key = {
            model_id, subgraph_def.worker_id};
        const SubgraphKey key = {model_id, subgraph_def.worker_id,
                                 subgraph_def.unit_subgraph_indices};

        if (interpreters_.find(interpreter_key) == interpreters_.end()) {
          BAND_REPORT_ERROR(error_reporter_,
                            "Subgraph logic created a subgraph for worker %d "
                            "that does not supports model %d",
                            subgraph_def.worker_id, model_id);
        } else {
          auto& interpreter = interpreters_[{model_id, subgraph_def.worker_id}];
          BandStatus status = interpreter->PrepareSubgraph(
              model->GetBackendModel(backend_type), subgraph_def.op_indices,
              subgraph_def.unit_subgraph_indices);
          if (status == kBandOk) {
            // Verify generated subgraphs
            BAND_ENSURE(error_reporter_, interpreter->HasSubgraph(key));
            const std::set<int> inputs =
                model_spec.GetPureInputTensors(subgraph_def.op_indices);
            const std::set<int> all_outputs =
                model_spec.GetOutputTensors(subgraph_def.op_indices);
            BAND_ENSURE(
                error_reporter_,
                std::equal(interpreter->GetInputs(key).begin(),
                           interpreter->GetInputs(key).end(), inputs.begin()));
            BAND_ENSURE(error_reporter_,
                        std::includes(all_outputs.begin(), all_outputs.end(),
                                      interpreter->GetOutputs(key).begin(),
                                      interpreter->GetOutputs(key).end()));
          }
        }
      }

      // Verify equality of all tensor pairs
      for (const SubgraphDef& lhs : subgraph_defs) {
        const std::pair<ModelId, WorkerId> lhs_interpreter_key = {
            model_id, lhs.worker_id};
        auto& lhs_interpreter = interpreters_[{model_id, lhs.worker_id}];
        const SubgraphKey lhs_key = {model_id, lhs.worker_id,
                                     lhs.unit_subgraph_indices};

        std::set<int> lhs_outputs{lhs_interpreter->GetOutputs(lhs_key).begin(),
                                  lhs_interpreter->GetOutputs(lhs_key).end()};

        for (const SubgraphDef& rhs : subgraph_defs) {
          const std::pair<ModelId, WorkerId> rhs_interpreter_key = {
              model_id, rhs.worker_id};
          auto& rhs_interpreter = interpreters_[{model_id, rhs.worker_id}];
          const SubgraphKey rhs_key = {model_id, rhs.worker_id,
                                       rhs.unit_subgraph_indices};
          if ((lhs.worker_id != rhs.worker_id) && (&lhs != &rhs)) {
            std::set<int> rhs_inputs{
                rhs_interpreter->GetInputs(rhs_key).begin(),
                rhs_interpreter->GetInputs(rhs_key).end()};

            std::set<int> common_tensors;
            std::set_intersection(
                lhs_outputs.begin(), lhs_outputs.end(), rhs_inputs.begin(),
                rhs_inputs.end(),
                std::inserter(common_tensors, common_tensors.end()));

            for (int common_tensor_index : common_tensors) {
              if (!(*lhs_interpreter->GetTensorView(lhs_key,
                                                    common_tensor_index) ==
                    *rhs_interpreter->GetTensorView(rhs_key,
                                                    common_tensor_index))) {
                BAND_LOG_PROD(BAND_LOG_ERROR, "%s %s %d != %s %s %d",
                              BandDeviceGetName(GetWorkerDevice(lhs.worker_id)),
                              lhs.ToString().c_str(), common_tensor_index,
                              BandDeviceGetName(GetWorkerDevice(rhs.worker_id)),
                              rhs.ToString().c_str(), common_tensor_index);
              }
            }
          }
        }
      }

      // todo: connect prev / next && unit indices

      // Initialize tensor ring buffer
      // Assumption: each backend model in Band::Model has the same input /
      // output tensor shapes
      {
        std::vector<std::shared_ptr<Interface::ITensor>> input_tensors;
        std::vector<std::shared_ptr<Interface::ITensor>> output_tensors;

        auto model_subgraph_key =
            GetLargestSubgraphKey(model_id, GetDeviceWorkerId(kBandCPU));
        Interface::IInterpreter* primary_interpreter =
            GetInterpreter(model_subgraph_key);

        for (int input_tensor : model_spec.input_tensors) {
          input_tensors.push_back(primary_interpreter->GetTensorView(
              model_subgraph_key, input_tensor));
        }

        for (int output_tensor : model_spec.output_tensors) {
          output_tensors.push_back(primary_interpreter->GetTensorView(
              model_subgraph_key, output_tensor));
        }

        const std::vector<int> input_indices{model_spec.input_tensors.begin(),
                                             model_spec.input_tensors.end()};
        const std::vector<int> output_indices{model_spec.output_tensors.begin(),
                                              model_spec.output_tensors.end()};

        model_input_buffer_.emplace(
            model->GetId(), std::make_unique<TensorRingBuffer>(
                                error_reporter_, input_tensors, input_indices));
        model_output_buffer_.emplace(
            model_id, std::make_unique<TensorRingBuffer>(
                          error_reporter_, output_tensors, output_indices));
      }
    }

    if (planner_->NeedProfile()) {
      latency_estimator_->ProfileModel(model_id);
    }

    return kBandOk;
  }
}

BandStatus Engine::UnregisterModel(Model* model) {
  if (!model) {
    BAND_LOG_INTERNAL(BAND_LOG_ERROR, "Failed to unregister null model.");
    return kBandError;
  }

  for (auto it = interpreters_.begin(); it != interpreters_.end();) {
    (it->first.second == model->GetId()) ? interpreters_.erase(it++) : (++it);
  }

  for (auto it = model_specs_.begin(); it != model_specs_.end();) {
    (it->first == model->GetId()) ? model_specs_.erase(it++) : (++it);
  }

  for (auto it = model_input_buffer_.begin();
       it != model_input_buffer_.end();) {
    (it->first == model->GetId()) ? model_input_buffer_.erase(it++) : (++it);
  }

  for (auto it = model_output_buffer_.begin();
       it != model_output_buffer_.end();) {
    (it->first == model->GetId()) ? model_output_buffer_.erase(it++) : (++it);
  }

  return kBandOk;
}

Tensor* Engine::CreateTensor(ModelId model_id, int tensor_index) {
  // TODO: What if there are multiple backends?
  SubgraphKey model_subgraph_key =
      GetLargestSubgraphKey(model_id, GetDeviceWorkerId(kBandCPU));

  if (Interface::IInterpreter* interpreter =
          GetInterpreter(model_subgraph_key)) {
    return new Tensor(
        interpreter->GetTensorView(model_subgraph_key, tensor_index).get());
  } else {
    return nullptr;
  }
}

std::vector<int> Engine::GetOutputTensorIndices(ModelId model_id) const {
  SubgraphKey model_subgraph_key =
      GetLargestSubgraphKey(model_id, GetDeviceWorkerId(kBandCPU));
  const Interface::IInterpreter* interpreter =
      GetInterpreter(model_subgraph_key);
  return interpreter ? interpreter->GetOutputs(model_subgraph_key)
                     : std::vector<int>();
}

std::vector<int> Engine::GetInputTensorIndices(ModelId model_id) const {
  SubgraphKey model_subgraph_key =
      GetLargestSubgraphKey(model_id, GetDeviceWorkerId(kBandCPU));
  const Interface::IInterpreter* interpreter =
      GetInterpreter(model_subgraph_key);
  return interpreter ? interpreter->GetInputs(model_subgraph_key)
                     : std::vector<int>();
}

size_t Engine::GetNumWorkers() const { return workers_.size(); }

BandDeviceFlags Engine::GetWorkerDevice(WorkerId id) const {
  if (id >= 0 && id < workers_.size()) {
    return workers_.at(id)->GetDeviceFlag();
  } else {
    BAND_REPORT_ERROR(error_reporter_, "Invalid worker id %d", id);
    return kBandNumDevices;
  }
}

BandStatus Engine::RequestSync(ModelId model_id, BandRequestOption options,
                               Tensors inputs, Tensors outputs) {
  JobId job_id = RequestAsync(model_id, options, inputs);
  if (job_id == -1) {
    return kBandError;
  } else {
    return Wait(job_id, outputs);
  }
}

BandStatus Engine::RequestSync(std::vector<ModelId> model_ids,
                               std::vector<BandRequestOption> options,
                               std::vector<Tensors> inputs,
                               std::vector<Tensors> outputs) {
  std::vector<JobId> job_ids = RequestAsync(model_ids, options, inputs);
  if (job_ids.size() > 0) {
    return kBandError;
  } else {
    return Wait(job_ids, outputs);
  }
}

JobId Engine::RequestAsync(ModelId model_id, BandRequestOption options,
                           Tensors inputs) {
  std::vector<Tensors> input_tensors;
  if (inputs.size()) {
    input_tensors.push_back(inputs);
  }
  auto job_ids = RequestAsync({model_id}, {options}, input_tensors);
  return job_ids.size() == 1 ? job_ids[0] : -1;
}

std::vector<JobId> Engine::RequestAsync(std::vector<ModelId> model_ids,
                                        std::vector<BandRequestOption> options,
                                        std::vector<Tensors> inputs) {
  std::vector<Job> jobs;

  if (model_ids.size() != options.size()) {
    BAND_REPORT_ERROR(error_reporter_,
                      "# Model requests (%llu) != # Worker ids (%llu)",
                      model_ids.size(), options.size());
    return {};
  }

  for (size_t i = 0; i < model_ids.size(); i++) {
    // TODO(BAND-33): explicit job life cycle
    Job job(model_ids[i]);
    job.require_callback = options[i].require_callback;

    int target_slo_us = options[i].slo_us;
    if (options[i].slo_scale != -1) {
      if (options[i].slo_scale <= 0) {
        BAND_REPORT_ERROR(error_reporter_,
                          "Specified slo_scale is invalid (%f <= 0)",
                          options[i].slo_scale);
        return {};
      }

      target_slo_us = GetWorst(model_ids[i]) * options[i].slo_scale;
    }

    // override, if `slo_us` is specified
    if (options[i].slo_us != -1) {
      target_slo_us = options[i].slo_us;
    }

    job.slo_us = target_slo_us;

    if (options[i].target_worker != -1) {
      Worker* target_worker = GetWorker(options[i].target_worker);
      if (target_worker == nullptr) {
        BAND_REPORT_ERROR(error_reporter_,
                          "Request assigned to invalid worker id (%d)",
                          options[i]);
        return {};
      }
      job.target_worker_id = options[i].target_worker;
    }

    if (i < inputs.size()) {
      int input_handle = model_input_buffer_[model_ids[i]]->Alloc();
      if (model_input_buffer_[model_ids[i]]->PutTensorsToHandle(
              inputs[i], input_handle) != kBandOk) {
        BAND_REPORT_ERROR(error_reporter_, "Input copy failure for model %d",
                          model_ids[i]);
        return {};
      }
      job.input_handle = input_handle;
      job.output_handle = model_output_buffer_[model_ids[i]]->Alloc();
    }

    jobs.push_back(job);
  }

  for (ModelId model_id : model_ids) {
  }
  return EnqueueBatch(jobs);
}

BandStatus Engine::Wait(JobId job_id, Tensors outputs) {
  std::vector<Tensors> output_tensors;
  if (outputs.size()) {
    output_tensors.push_back(outputs);
  }
  return Wait(std::vector<JobId>({job_id}), output_tensors);
}

BandStatus Engine::Wait(std::vector<JobId> job_ids,
                        std::vector<Tensors> outputs) {
  planner_->Wait(job_ids);
  for (size_t i = 0; i < outputs.size(); i++) {
    BAND_ENSURE_STATUS(GetOutputTensors(job_ids[i], outputs[i]));
  }
  return kBandOk;
}

BandStatus Engine::GetOutputTensors(JobId job_id, Tensors outputs) {
  Job job = planner_->GetFinishedJob(job_id);

  if (outputs.empty() || job_id == -1) {
    BAND_REPORT_ERROR(error_reporter_,
                      "Invalid job id / num outputs to copy: (%d, %d)", job_id,
                      outputs.size());
    return kBandError;
  }

  // Not finished or invalidated
  if (job.job_id == -1) {
    return kBandError;
  }

  if (job.output_handle == -1) {
    BAND_REPORT_ERROR(error_reporter_, "Invalid output handle : %d",
                      job.output_handle);
    return kBandError;
  }

  if (model_output_buffer_.find(job.model_id) == model_output_buffer_.end()) {
    BAND_REPORT_ERROR(error_reporter_, "Invalid model id : %d", job.model_id);
    return kBandError;
  }

  BAND_ENSURE_STATUS(model_output_buffer_.at(job.model_id)
                         ->GetTensorsFromHandle(outputs, job.output_handle));
  return kBandOk;
}

void Engine::SetOnEndRequest(
    std::function<void(int, BandStatus)> on_end_request) {
  planner_->SetOnEndRequest(on_end_request);
}

BandStatus Engine::Init(const RuntimeConfig& config) {
  planner_ = std::make_unique<Planner>(this);

  BAND_ENSURE_STATUS(planner_->Init(config.planner_config));

  {
    model_config_ = config.model_config;

    if (planner_->NeedProfile()) {
      latency_estimator_ = std::make_unique<LatencyEstimator>(this);
      BAND_ENSURE_STATUS(latency_estimator_->Init(config.profile_config));
    }

    const BandCPUMaskFlags cpu_mask =
        static_cast<BandCPUMaskFlags>(config.cpu_mask);
    auto cpu_mask_set = BandCPUMaskGetSet(cpu_mask);

    BAND_LOG_INTERNAL(BAND_LOG_INFO, "Set affinity to %s cores.",
                      BandCPUMaskGetName(cpu_mask));

    BAND_ENSURE_STATUS(SetCPUThreadAffinity(cpu_mask_set));
  }

  // Search for all available backends, devices
  std::set<BandDeviceFlags> valid_devices;
  auto valid_backends = BackendFactory::GetAvailableBackends();
  for (auto backend : valid_backends) {
    auto backend_devices =
        BackendFactory::GetBackendUtil(backend)->GetAvailableDevices();
    valid_devices.insert(backend_devices.begin(), backend_devices.end());
  }

  auto& potential_workers = config.worker_config.workers;
  for (int i = 0; i < potential_workers.size(); i++) {
    BandDeviceFlags device_flag = potential_workers[i];
    if (valid_devices.find(device_flag) != valid_devices.end()) {
      std::unique_ptr<Worker> worker;
      if (planner_->GetWorkerType() == kBandGlobalQueue) {
        worker = std::make_unique<GlobalQueueWorker>(this, workers_.size(),
                                                     device_flag);
      } else {
        worker = std::make_unique<DeviceQueueWorker>(this, workers_.size(),
                                                     device_flag);
      }
      if (worker->Init(config.worker_config) != kBandOk) {
        error_reporter_->Report("Worker::Init() failed for worker : %s.",
                                BandDeviceGetName(device_flag));
        return kBandError;
      }

      BAND_LOG_INTERNAL(BAND_LOG_INFO, "%s worker is created.",
                        BandDeviceGetName(device_flag));
      worker->Start();
      workers_.push_back(std::move(worker));
    } else {
      BAND_LOG_INTERNAL(BAND_LOG_WARNING, "%s worker is not created.",
                        BandDeviceGetName(device_flag));
    }
  }

  return kBandOk;
}

Engine::Engine(ErrorReporter* error_reporeter) : Context(error_reporeter) {}

void Engine::UpdateWorkerWaitingTime() const {
  for (WorkerId worker_id = 0; worker_id < workers_.size(); worker_id++) {
    workers_waiting_[worker_id] = workers_[worker_id]->GetWaitingTime();
  }
}

const WorkerWaitingTime& Engine::GetWorkerWaitingTime() const {
  return workers_waiting_;
}

std::set<int> Engine::GetIdleWorkers() const {
  std::set<int> idle_workers;

  for (WorkerId worker_id = 0; worker_id < workers_.size(); worker_id++) {
    if (!workers_[worker_id]->HasJob()) {
      idle_workers.insert(worker_id);
    }
  }
  return idle_workers;
}

SubgraphKey Engine::GetLargestSubgraphKey(ModelId model_id,
                                          WorkerId worker_id) const {
  auto interpreter_it = interpreters_.find({model_id, worker_id});
  if (interpreter_it != interpreters_.end()) {
    return interpreter_it->second->GetLargestSubgraphKey();
  } else {
    return SubgraphKey();
  }
}

const ModelSpec* Engine::GetModelSpec(ModelId model_id) const {
  if (model_specs_.find(model_id) == model_specs_.end()) {
    return nullptr;
  } else {
    return &model_specs_.at(model_id);
  }
}

WorkerId Engine::GetModelWorker(ModelId model_id) const {
  return planner_->GetModelWorkerMap()[model_id];
}

bool Engine::IsEnd(const SubgraphKey& key) const {
  const ModelSpec* model_spec = GetModelSpec(key.GetModelId());
  // check whether key has the last unit subgraph
  return model_spec &&
             (key.GetUnitIndices().find(model_spec->num_unit_subgraphs - 1) !=
              key.GetUnitIndices().end()) ||
         (key.GetUnitIndices().size() == 0);
}

bool Engine::HasSubgraph(const SubgraphKey& key) const {
  auto interpreter_it =
      interpreters_.find({key.GetModelId(), key.GetWorkerId()});
  return interpreter_it != interpreters_.end() &&
         interpreter_it->second->HasSubgraph(key);
}

BandStatus Engine::Invoke(const SubgraphKey& key) {
  auto interpreter_it =
      interpreters_.find({key.GetModelId(), key.GetWorkerId()});
  if (interpreter_it != interpreters_.end()) {
    return interpreter_it->second->InvokeSubgraph(key);
  } else {
    BAND_LOG_INTERNAL(BAND_LOG_WARNING, "Failed to find a subgraph key");
    return kBandError;
  }
}

std::pair<SubgraphKey, int64_t> Engine::GetShortestLatency(
    int model_id, std::set<int> resolved_tensors, int64_t start_time,
    const std::map<WorkerId, int64_t>& worker_waiting,
    SubgraphKey preceded_subgraph_index) const {
  BAND_NOT_IMPLEMENTED;
  return {};
}

std::pair<std::vector<SubgraphKey>, int64_t>
Engine::GetShortestLatencyWithUnitSubgraph(
    int model_id, int start_unit_idx,
    const std::map<WorkerId, int64_t>& worker_waiting) const {
  std::pair<std::vector<SubgraphKey>, int64_t> result;
  BAND_NOT_IMPLEMENTED;
  return result;
}

std::pair<std::vector<SubgraphKey>, int64_t>
Engine::GetSubgraphWithShortestLatency(
    Job& job, const std::map<WorkerId, int64_t>& worker_waiting) const {
  std::pair<std::vector<SubgraphKey>, int64_t> result;
  BAND_NOT_IMPLEMENTED;
  return result;
}

SubgraphKey Engine::GetSubgraphIdxSatisfyingSLO(
    Job& job, const std::map<WorkerId, int64_t>& worker_waiting,
    const std::set<WorkerId>& idle_workers) const {
  BAND_NOT_IMPLEMENTED;
  return {};
}

void Engine::UpdateLatency(const SubgraphKey& key, int64_t latency) {
  if (latency_estimator_) latency_estimator_->UpdateLatency(key, latency);
}

int64_t Engine::GetProfiled(const SubgraphKey& key) const {
  return latency_estimator_ ? latency_estimator_->GetProfiled(key) : 0;
}

int64_t Engine::GetExpected(const SubgraphKey& key) const {
  return latency_estimator_ ? latency_estimator_->GetExpected(key) : 0;
}

int64_t Engine::GetWorst(ModelId model_id) const {
  // requires nullity check for schedulers without profile
  return latency_estimator_ ? latency_estimator_->GetWorst(model_id) : 0;
}

void Engine::Trigger() { planner_->Trigger(); }

int Engine::EnqueueRequest(Job job, bool push_front) {
  return planner_->EnqueueRequest(job, push_front);
}

std::vector<int> Engine::EnqueueBatch(std::vector<Job> jobs, bool push_front) {
  return planner_->EnqueueBatch(jobs, push_front);
}

void Engine::PrepareReenqueue(Job& job) { planner_->PrepareReenqueue(job); }

void Engine::EnqueueFinishedJob(Job& job) { planner_->EnqueueFinishedJob(job); }

const Worker* Engine::GetWorker(WorkerId id) const {
  if (id >= 0 && id < workers_.size()) {
    return workers_.at(id).get();
  } else {
    return nullptr;
  }
}

Worker* Engine::GetWorker(WorkerId id) {
  if (id >= 0 && id < workers_.size()) {
    return workers_[id].get();
  } else {
    return nullptr;
  }
}

BandStatus Engine::TryCopyInputTensors(const Job& job) {
  // Skip all tensor communication for compute only case.
  if (job.input_handle < 0) {
    return kBandOk;
  }

  // TODO: Subgraph execution
  const SubgraphKey& key = job.subgraph_key;
  auto interpeter = GetInterpreter(job.subgraph_key);
  std::set<int> unresolved_tensors(interpeter->GetInputs(key).begin(),
                                   interpeter->GetInputs(key).end());

  // // Intermediate tensor communication
  // for (auto subgraph_it = job.previous_subgraph_indices.cbegin();
  //      subgraph_it != job.previous_subgraph_indices.cend();
  //      ++subgraph_it) {
  //   int preceded_subgraph_index = *subgraph_it;
  //   Subgraph *preceded_subgraph =
  //       interpreter->subgraph(preceded_subgraph_index);

  //   for (int tensor_index : preceded_subgraph->outputs()) {
  //     if (unresolved_tensors.find(tensor_index) !=
  //     unresolved_tensors.end())
  //     {
  //       const ITensorView *src = preceded_subgraph->tensor(tensor_index);
  //       ITensorView *dst = subgraph->tensor(tensor_index);

  //       if (ITensorDataCopy(src, dst) == kBandError) {
  //         BAND_REPORT_ERROR(GetErrorReporter(),
  //                           "Tensor data copy failure from %s to %s",
  //                           src->name, dst->name);
  //         return kBandError;
  //       }

  //       unresolved_tensors.erase(tensor_index);
  //     }
  //   }
  // }

  if (model_input_buffer_.find(job.model_id) == model_input_buffer_.end()) {
    BAND_LOG_INTERNAL(BAND_LOG_ERROR,
                      "Failed to find input tensor ring buffer for model %d",
                      job.model_id);
    return kBandError;
  }

  auto input_buffer = model_input_buffer_[job.model_id].get();

  // Copy model input
  for (auto tensor_it = unresolved_tensors.begin();
       tensor_it != unresolved_tensors.end();) {
    int tensor_index = *tensor_it;
    if (input_buffer->IsTensorIndexValid(tensor_index)) {
      if (input_buffer->GetTensorFromHandle(
              interpeter->GetTensorView(key, tensor_index).get(), tensor_index,
              job.input_handle) != kBandOk) {
        BAND_LOG_INTERNAL(BAND_LOG_ERROR,
                          "Failed to copy input tensor %d for model %d",
                          tensor_index, job.model_id);
        return kBandError;
      }
      tensor_it = unresolved_tensors.erase(tensor_it);
    } else {
      ++tensor_it;
    }
  }

  if (unresolved_tensors.empty()) {
    return kBandOk;
  } else {
    return kBandError;
  }

  return kBandOk;
}

BandStatus Engine::TryCopyOutputTensors(const Job& job) {
  // TODO: Subgraph execution

  // Compute only.
  if (job.output_handle < 0) {
    return kBandOk;
  }

  const SubgraphKey& key = job.subgraph_key;
  auto interpeter = GetInterpreter(job.subgraph_key);

  if (model_output_buffer_.find(job.model_id) == model_output_buffer_.end()) {
    BAND_LOG_INTERNAL(BAND_LOG_ERROR,
                      "Failed to find output tensor ring buffer for model %d",
                      job.model_id);
    return kBandError;
  }

  auto output_buffer = model_output_buffer_[job.model_id].get();
  for (int tensor_index : interpeter->GetOutputs(key)) {
    if (output_buffer->IsTensorIndexValid(tensor_index)) {
      if (output_buffer->PutTensorToHandle(
              interpeter->GetTensorView(key, tensor_index).get(), tensor_index,
              job.output_handle) != kBandOk) {
        BAND_LOG_INTERNAL(BAND_LOG_ERROR,
                          "Failed to copy output tensor %d for model %d",
                          tensor_index, job.model_id);
        return kBandError;
      }
    }
  }

  return kBandOk;
}

WorkerId Engine::GetDeviceWorkerId(BandDeviceFlags flag) const {
  for (WorkerId worker_id = 0; worker_id < workers_.size(); worker_id++) {
    if (workers_[worker_id]->GetDeviceFlag() == flag) {
      return worker_id;
    }
  }
  BAND_LOG_INTERNAL(BAND_LOG_WARNING, "Failed to find a worker for %s",
                    BandDeviceGetName(flag));
  return -1;
}

Interface::IInterpreter* Engine::GetInterpreter(const SubgraphKey& key) {
  auto it = interpreters_.find({key.GetModelId(), key.GetWorkerId()});
  return it != interpreters_.end() ? it->second.get() : nullptr;
}

const Interface::IInterpreter* Engine::GetInterpreter(
    const SubgraphKey& key) const {
  auto it = interpreters_.find({key.GetModelId(), key.GetWorkerId()});
  return it != interpreters_.end() ? it->second.get() : nullptr;
}
}  // namespace Band