import torch
from torch import nn

class Model(nn.Module):

    def __init__(self, batch, height, width, channel_in):
        super().__init__()
        
        self.net = nn.Sequential(
                nn.Conv2d(channel_in, 6, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                )
        self.linear = nn.Linear(int(height*width*6/4), 8)
        self.act = nn.Softmax(1)

    def forward(self, X):
        out = self.net(X.permute(0,3,2,1)).flatten(1)

        return self.act(self.linear(out))

if __name__=='__main__':
    
    # create dummy input
    B, H, W, C = 20, 64, 256, 3
    X = torch.randn(B, H, W, C)
    
    # initialize model
    model = Model(B,H,W,C)

    # run model
    output = model(X)
    
    # debug output
    print(output.shape)
    
    # save model
    torch.save(model.state_dict(), "./weights/weights.pth")
    
