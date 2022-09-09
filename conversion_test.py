import torch
import sys
sys.path.insert(0,'./torch-model/')
from model import Model
import onnxruntime as ort
import numpy as np

if __name__ == '__main__':
    
    # create dummy input
    B, H, W, C = 1, 64, 256, 3
    X = torch.full((B, H, W, C),1.0)
   
    # load torch model
    torch_weights = torch.load('./torch-model/weights/weights.pth')
    torch_model = Model(B, H, W, C)
    torch_model.load_state_dict(torch_weights)


    # load onnx model
    onnx_model = ort.InferenceSession("onnx-model/onnx_model.onnx")
    
    # run inference torch
    torch_model.eval()
    torch_output = torch_model(X)

    # run inference onnx
    # onnx_output = onnx_model.run(None, {'input': X.numpy().astype(np.float32)})[0]
    onnx_output = onnx_model.run(None, {'input': X.numpy()})[0]
    
    # debug outputs
    print(f'torch output: {torch_output[0,:]}')
    print(f'onnx output: {onnx_output[0,:]}')

