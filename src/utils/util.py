#
# Copyright 2022 DMetaSoul
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys

import onnx
import torch
import numpy as np
import onnxruntime

def validate_onnx_model(model, onnx_path, dummy=None, device='cpu', print_model=False, rtol=1e-02, atol=1e-03):
    # Check that the exported model is well formed
    print("Checking ONNX model format...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("Checking done!")

    # Print a human readable representation of the graph
    if print_model:
        print(onnx.helper.printable_graph(onnx_model.graph))

    # Verify that ONNX Runtime and PyTorch are computing the same value for the network
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    #print([x.name for x ort_session.get_inputs()])

    input_names = model.input_names
    output_names = model.output_names
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        tensor_inputs = model.get_dummy_inputs(dummy=dummy, batch_size=1, return_tensors="pt", device=device)
        torch_outs = model(**tensor_inputs)
        torch_out_keys = set(torch_outs.keys())

    print("Validating ONNX model...")
    ort_inputs = {k:tensor_inputs[k].cpu().numpy() for k in input_names}
    ort_out_keys = set(output_names)
    ort_outs = ort_session.run(output_names, ort_inputs)
    #ort_outs = ort_session.run(None, ort_inputs)
    if not ort_out_keys.issubset(torch_out_keys):
        print(f"\t-[x] ONNX model output names {ort_out_keys} do not match reference model {ort_out_keys}")
        raise ValueError("Model validation failed!")
    else:
        print(f"\t-[✓] ONNX model output names match reference model ({ort_out_keys})")

    for name, ort_value in zip(output_names, ort_outs):
        print(f'\t- Validating ONNX Model output "{name}":')
        ref_value = torch_outs[name].numpy()

        if not ort_value.shape == ref_value.shape:
            print(f"\t\t-[x] shape {ort_value.shape} doesn't match {ref_value.shape}")
            raise ValueError("Model validation failed!")
        else:
            print(f"\t\t-[✓] {ort_value.shape} matches {ref_value.shape}")

        if not np.allclose(ref_value, ort_value, atol=atol, rtol=rtol):
            print(f"\t\t-[x] values not close enough (atol: {atol}, rtol: {rtol})")
            raise ValueError("Model validation failed!")
        else:
            print(f"\t\t-[✓] all values close (atol: {atol}, rtol: {rtol})")

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def onnx_export(model, onnx_path, dummy=None, device='cpu', onnx_version=11):
    model.eval()
    model.to(device)

    with torch.no_grad():
        model_inputs = model.get_dummy_inputs(dummy=dummy, device=device)
        assert isinstance(model_inputs, dict), "The model dummy inputs must be a dict!"

        dynamic_axes = {}
        dynamic_axes.update(model.input_axes)
        dynamic_axes.update(model.output_axes)

        torch.onnx.export(model,
            args=(model_inputs,),  # https://pytorch.org/docs/stable/onnx.html#torch.onnx.export
            f=onnx_path,
            input_names=model.input_names,
            output_names=model.output_names,
            dynamic_axes=dynamic_axes,
            #verbose=True,
            do_constant_folding=True,  # whether to execute constant folding for optimization
            export_params=True,        # store the trained parameter weights inside the model file
            opset_version=onnx_version
        )

def model_exporter(model, export_path,
        onnx_name='model.onnx', config_name='export_config.json', device='cpu', 
        validate=True, save_pretrained=True, dummy=None, onnx_version=11):
    # check model
    attr_names = ['input_names', 'output_names', 'input_axes', 'output_axes']
    method_names = ['get_dummy_inputs', 'save', 'forward']
    for name in attr_names:
        if not hasattr(model, name):
            raise Exception(f"Model doesn't have attribute {name}")
    for name in method_names:
        if not hasattr(model, name):
            raise Exception(f"Model doesn't have method {name}")

    main_dir = os.path.join(export_path, 'model_onnx')
    preprocess_dir = os.path.join(export_path, 'model_conf')
    onnx_path = os.path.join(main_dir, onnx_name)
    config_path = os.path.join(preprocess_dir, config_name)
    os.makedirs(main_dir, exist_ok=True)
    os.makedirs(preprocess_dir, exist_ok=True)

    # save transformer model
    if save_pretrained:
        model.save(preprocess_dir)

    ## export model via onnx
    onnx_export(model, onnx_path, dummy=dummy, device=device, onnx_version=onnx_version)

    if validate:
        validate_onnx_model(model, onnx_path, print_model=True, device=device)


    return model.input_names, model.output_names
