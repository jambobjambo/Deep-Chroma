# 
#   Deep Chroma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import Tensor
from torch.nn import Linear, Module, ReLU, Sequential, Tanh
from torch.nn.functional import interpolate
from torchvision.models import resnet34
from torchplasma.conversion import linear_to_srgb, rgb_to_xyz, srgb_to_linear, xyz_to_rgb
from torchplasma.filters import tone_curve
from torchsummary import summary
from typing import Tuple

# Deep Chromatic Adaptation

class DeepChroma (Module):

    def __init__ (self):
        super(DeepChroma, self).__init__()
        # Model
        self.model = resnet34(pretrained=True, progress=True)
        in_features = self.model.fc.in_features
        self.model.fc = Sequential(
            Linear(in_features, 1024),
            ReLU(inplace=True),
            Linear(1024, 256),
            ReLU(inplace=True),
            Linear(256, 64),
            ReLU(),
            Linear(64, 17),
            Tanh()
        )

    def forward (self, input: Tensor) -> Tensor:
        inverse_tone_curve, adaptation, forward_tone_curve = self.weights(input)
        result = self.adapt(input, inverse_tone_curve, adaptation, forward_tone_curve)
        return result

    def weights (self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute the chromatic adaptation weights for a given image.

        Parameters:
            input (Tensor): Input image with shape (N,3,H,W) in range [-1., 1.].

        Returns:
            tuple: Tuple containing inverse tone curve controls, chromatic adaptation matrix, and forward tone curve controls.
        """
        input = interpolate(input, size=(512, 512), mode="bilinear", align_corners=False)
        weights = self.model(input)
        inverse_tone_curve = weights[:,:4]
        adaptation = weights[:,4:13].view(-1, 3, 3)
        forward_tone_curve = weights[:,13:]
        return inverse_tone_curve, adaptation, forward_tone_curve

    def adapt (self, input: Tensor, inverse_tone_curve: Tensor, adaptation: Tensor, forward_tone_curve: Tensor) -> Tensor:
        """
        Apply the chromatic adaptation model to a given image.

        Parameters:
            input (Tensor): Input image with shape (N,3,H,W) in range [-1., 1.].
            inverse_tone_curve (Tensor): Inverse camera tone curve with shape (N,4) in range [-1., 1.].
            adaptation (Tensor): Chromatic adaptation matrix in XYZ space with shape (N,3,3).
            forward_tone_curve (Tensor): Forward camera tone curve with shape (N,4) in range [-1., 1.].

        Returns:
            Tensor: Color balanced image with shape (N,3,H,W) in range [-1., 1.].
        """
        # Apply inverse tone curve
        linear = tone_curve(input, inverse_tone_curve)
        linear = srgb_to_linear(linear)
        # Apply chromatic adaptation transform
        xyz = rgb_to_xyz(linear)
        xyz = adaptation @ xyz.flatten(start_dim=2)
        xyz = xyz.view_as(input)
        linear = xyz_to_rgb(xyz)
        # Apply forward tone curve
        result = linear_to_srgb(linear)
        result = tone_curve(result, forward_tone_curve)
        result = result.clamp(min=-1., max=1.)
        return result

    
if __name__ == "__main__":
    model = DeepChroma()
    summary(model, (3, 1024, 1024), batch_size=8)