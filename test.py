# 
#   Deep Chroma
#   Copyright (c) 2020 Homedeck, LLC.
#

from argparse import ArgumentParser
from PIL import Image
from torch import cat, device as get_device, set_grad_enabled
from torch.cuda import is_available as cuda_available
from torch.jit import load
from torchvision.transforms import Compose, Normalize, Resize, ToPILImage, ToTensor

# Parse arguments
parser = ArgumentParser(description="Deep Color: Test")
parser.add_argument("--model", type=str, default="deep_color.pt", help="Path to trained model")
parser.add_argument("--input", type=str, required=True, help="Path to input image")
args = parser.parse_args()

# Load model
device = get_device("cuda:0") if cuda_available() else get_device("cpu")
model = load(args.model, map_location=device).to(device)
set_grad_enabled(False)

# Transforms
to_tensor = Compose([
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
to_image = Compose([
    Normalize(mean=[-1., -1., -1.], std=[2., 2., 2.]),
    ToPILImage()
])

# Load image
input = Image.open(args.input)
input = to_tensor(input).unsqueeze(dim=0).to(device)

# Run forward
inverse_tone_curve, adaptation, forward_tone_curve = model.weights(input)
result = model.adapt(input, inverse_tone_curve, adaptation, forward_tone_curve)

# Output
to_image(result.cpu().squeeze(dim=0)).save(f"result.jpg")