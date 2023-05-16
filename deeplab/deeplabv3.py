import torch

class DeepLabV3(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        torchvision_version = 'pytorch/vision:v0.14.1'
        self.n_classes = num_classes
        
        self.main = torch.hub.load(torchvision_version, 'deeplabv3_resnet50', num_classes=num_classes, weights=None)
    
    def forward(self, x):
        input_channels = x.shape[1]
        assert input_channels in (1, 3)
        if input_channels == 1:
            x = x.expand(-1, 3, -1, -1)
        out = self.main(x)
        return out['out']
