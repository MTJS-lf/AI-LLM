import onnx
from onnx import numpy_helper
import onnxruntime
onnx_path='model_onnx/model_onnx/model.onnx'
model = onnx.load(onnx_path)
for initializer in model.graph.initializer:
        W= numpy_helper.to_array(initializer)
        print(initializer.name,W.shape)
        print(W)

