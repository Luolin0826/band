import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np

from proto import ModelDescriptor
from backend.util import (
    convert_dtype,
    convert_shape,
    get_bytes_size
)

class TensorRTBackend(object):
    def __init__(self, gpus=[0]):
        self._bindings = None
        self.devices = [cuda.Device(gpu) for gpu in gpus]
        print(f'Server has been initialized to use GPU({gpus})')
        # self.device_contexts = [device.make_context() for device in self.devices]
        self.engines = dict()
        self.contexts = dict()
        self.inputs = dict()
        self.outputs = dict()
        self.allocations = dict()
        
    @property
    def bindings(self):
        return self._bindings

    @property
    def bindings(self, value):
        self._bindings = value

    def load_model(self, model):
        model_id = model['id']
        model_name = model['name']
        model_path = model['path']
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(TRT_LOGGER, namespace="")
        runtime = trt.Runtime(TRT_LOGGER)
        with open(model_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
            self.engines[model_name] = engine
        
        context = self.engines[model_name].create_execution_context()
        self.contexts[model_name] = context
        assert self.engines[model_name]
        assert self.contexts[model_name]
        
        inputs, outputs = self.alloc_buf(model_name, self.engines[model_name], self.contexts[model_name])
        return ModelDescriptor(
            name=model_name,
            id=model_id,
            num_ops=len(inputs) + len(outputs),
            num_tensors=len(inputs) + len(outputs),
            tensor_types=[],
            input_tensor_indices=[input['index'] for input in inputs],
            output_tensor_indices=[output['index'] for output in outputs],
            op_input_tensors=[],
            op_output_tensors=[],
        )

    def inference(self, model_name, input_image):
        cuda.memcpy_htod(self.inputs[model_name][0]['allocation'], input_image)
        self.contexts[model_name].execute_v2(self.allocations[model_name])
        for i in range(len(self.outputs[model_name])):
            cuda.memcpy_dtoh(self.outputs[model_name][i]['host_allocation'], self.outputs[model_name][i]['allocation'])
        # TODO

    def alloc_buf(self, model_name, engine, context):
        self.inputs[model_name] = []
        self.outputs[model_name] = []
        self.allocations[model_name] = []

        for i in range(engine.num_bindings):
            is_input = False
            if engine.binding_is_input(i):
                is_input = True
            name = engine.get_binding_name(i)
            dtype = engine.get_binding_dtype(i)
            shape = context.get_binding_shape(i)

            if is_input:
                if shape[0] < 0:
                    assert engine.num_optimization_profiles > 0
                    profile_shape = engine.get_profile_shape(0, name)
                    assert len(profile_shape) == 3
                    context.set_binding_shape(i, profile_shape[2])
                    shape = context.get_binding_shape(i)
            size = get_bytes_size(dtype)
            for s in shape:
                size *= s

            allocation = cuda.mem_alloc(size)
            host_allocation = None if is_input else np.zeros(shape, trt.nptype(dtype))
            binding = {
                "index": i,
                "name": name,
                "dtype": convert_dtype(dtype),
                "shape": convert_shape(shape),
                "allocation": allocation,
                "host_allocation": host_allocation,
            }
            self.allocations[model_name].append(allocation)
            if engine.binding_is_input(i):
                self.inputs[model_name].append(binding)
            else:
                self.outputs[model_name].append(binding)

        assert len(self.inputs[model_name]) > 0
        assert len(self.outputs[model_name]) > 0
        assert len(self.allocations[model_name]) > 0
        return [self.inputs[model_name], self.outputs[model_name]]

    def input_spec(self, model_name):
        specs = []
        for i in self.inputs[model_name]:
            specs.append((i['shape'], i['dtype']))
        return specs

    def output_spec(self, model_name):
        specs = []
        for o in self.outputs[model_name]:
            specs.append((o['shape'], o['dtype']))
        return specs
