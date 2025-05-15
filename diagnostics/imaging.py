"""
Imaging (MRI) diagnosis using fine-tuned HRNet for ADRD.
"""
import torch
from torchvision import transforms
from PIL import Image
import os

MODEL_PATH = os.environ.get("HRNET_MODEL_PATH", os.path.join(os.path.dirname(__file__), "..", "models", "hrnet_ad.pt"))

class SimpleHRNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Dummy model for placeholder; replace with actual HRNet import
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(256*256*3, 1)
        )
    def forward(self, x):
        return self.net(x)

def load_hrnet_model(model_path=MODEL_PATH):
    # Replace with actual HRNet model import and loading
    model = SimpleHRNet()
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
        except Exception:
            pass  # Ignore for stub/demo
    model.eval()
    return model

def preprocess_image(image_path: str):
    """
    Preprocess MRI image for HRNet inference.
    """
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return transform(img).unsqueeze(0)

def imaging_ad_prediction(data: dict) -> bool:
    """
    Run HRNet inference on MRI image for AD prediction.
    Args:
        data (dict): Must contain 'image_path'.
    Returns:
        bool: True if AD detected, False otherwise.
    """
    image_path = data.get("image_path")
    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    model = load_hrnet_model()
    input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
    return prob > 0.5
