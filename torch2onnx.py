import torch
import sys
sys.path.insert(0,'./torch-model/')
from model import Model

if __name__ == '__main__':
    
    # create dummy input
    B, H, W, C = 20, 64, 256, 3
    X = torch.full((B, H, W, C),1.0).detach()
   
    # load torch model
    torch_weights = torch.load('./torch-model/weights/weights.pth')
    torch_model = Model(B, H, W, C)

    # export to onnx
    torch.onnx.export(torch_model,                  # model
                     X,                             # dummy input
                     "onnx-model/onnx_model.onnx",  # save path
                     export_params=True,             # store the trained weights inside file model
                     opset_version=12,              # onnx version model
                     do_constant_folding=True,      # constant folding optimization
                     input_names = ['input'],       # models inputs names
                     output_names = ['output'],
                     dynamic_axes={'input': {0: 'batch_size'}, # variable length
                                    'output': {0: 'batch_size'}}
                     )
