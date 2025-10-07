from torchvision.models import vgg16

import torch
import torch.nn.functional as F

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, layers=[3, 8, 15, 22], weight=1.0):
        super().__init__()
        vgg = vgg16(pretrained=True).features.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.layers = layers
        self.weight = weight

    def forward(
            self, 
            x: torch.Tensor, 
            y: torch.Tensor
        ):
        """Compute perceptual loss

        Args:
            x (torch.Tensor): model output
            y (torch.Tensor): ground truth label

        Returns:
            _type_: _description_
        """
        feats_x, feats_y = x, y
        loss = 0.0
        for i, layer in enumerate(self.vgg):
            feats_x = layer(feats_x)
            feats_y = layer(feats_y)
            if i in self.layers:
                loss += F.mse_loss(feats_x, feats_y)
        return self.weight * loss

