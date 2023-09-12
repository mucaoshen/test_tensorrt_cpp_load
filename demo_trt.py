import torch
import yaml
import tensorrt as trt
import json
from collections import OrderedDict, namedtuple
import numpy as np
from pathlib import Path
import re

def yaml_load(file='data.yaml', append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        dict: YAML data and file name.
    """
    import yaml
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)

        # Add YAML filename to dict and return
        return {**yaml.safe_load(s), 'yaml_file': str(file)} if append_filename else yaml.safe_load(s)



def export_engine_reg(ori_path, workspace=1, verbose=False):
    import tensorrt as trt
    print("trt version: ", trt.__version__)
    metadata = {
        'description': "test model tensorrt",
        'author': "jack huang",
        'batch': "None",
        'inputsz': {"images": [-1, 1, 36, 450]},
        'outputsz': {'output0': [-1, 57, 14665]},
        'names': "",
        'input_names': ["images"],
        'output_names': ['output0'],
    }

    f = str(Path(ori_path).with_suffix(".engine"))
    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    # config.max_workspace_size = workspace * 1 << 30t # this function is deprecated after trt 8.4
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace<<30)
    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(ori_path):
        raise RuntimeError(f'failed to load onnx file: {ori_path}')
    
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    for inp in inputs:
        print(f"TensorRT input '{inp.name}' with shape{inp.shape} {inp.dtype}")
    for out in outputs:
        print(f"TensorRT input '{out.name}' with shape{out.shape} {out.dtype}")
    profile = builder.create_optimization_profile()
    for inp in inputs:
        profile.set_shape(inp.name, (1, 1, 36, 450), (4, 1, 36, 450), (8, 1, 36, 450))
    config.add_optimization_profile(profile)

    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
        # meta = json.dumps(metadata)
        # t.write(len(meta).to_bytes(4, 'little', signed=True))
        # t.write(meta.encode())
        t.write(engine.serialize())
    return f, None

def get_engine(engine_path):

    device = torch.device('cuda:0')
    logger = trt.Logger(trt.Logger.INFO)

    with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
        # meta_len = int.from_bytes(f.read(4), byteorder='little')
        # metadata = json.loads(f.read(meta_len).decode('utf-8'))
        model = runtime.deserialize_cuda_engine(f.read())

    context = model.create_execution_context()
    bindings = OrderedDict()
    input_names = []
    output_names = []

    fp16 = False
    dynamic = False
    Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))

    for i in range(model.num_bindings):
        name = model.get_binding_name(i)
        dtype = trt.nptype(model.get_binding_dtype(i))

        if model.binding_is_input(i):
            if -1 in tuple(model.get_binding_shape(i)):
                dynamic = True
                context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
            
            if dtype == np.float16:
                fp16 = True
            input_names.append(name)

        else:
            output_names.append(name)
        shape = tuple(context.get_binding_shape(i))
        im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
        bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))

    binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
    batch_size = bindings['images'].shape[0]
    
    if isinstance(metadata, (str, Path)) and Path(metadata).exists():
        metadata = yaml_load(metadata)
    
    # if metadata:
    #     for k, v in metadata.items():
    #         if k in ('stride', 'batch'):
    #             metadata[k] = int(v)
    #         elif k in ('imgsz', 'names', 'kpt_shape') and isinstance(v, str):
    #             metadata[k] = eval(v)
        
    #     stride = metadata['stride']
    #     task = metadata['task']
    #     batch = metadata['batch']
    #     imgsz = metadata['imgsz']
    #     names = metadata['names']
    #     kpt_shape = metadata.get('kpt_shape')
    return model, bindings, context, binding_addrs, input_names, output_names, fp16

if __name__=='__main__':
    onnx_path = "ResNet34_trackerOCR_36_450_20230627_half.onnx"
    f, _ = export_engine_reg(onnx_path)
    model = get_engine(f)
