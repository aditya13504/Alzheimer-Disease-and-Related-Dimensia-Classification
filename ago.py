from typing import Dict, Any
import torch
from torchvision import transforms
from PIL import Image
import os
import git

# 1. Clinical Evaluation
def clinical_evaluation(data: Dict[str, Any]) -> bool:
    score = data.get("symptom_score", 0)
    return score > 5  # simple threshold

# 2. Cognitive Testing
def cognitive_testing(data: Dict[str, Any]) -> bool:
    test_score = data.get("cognitive_score", 30)
    return test_score < 24  # MMSE/MoCA cutoff

# 3. Laboratory Tests
def laboratory_test(data: Dict[str, Any]) -> bool:
    amyloid = data.get("amyloid_beta", 1000)
    tau = data.get("tau", 100)
    return tau > 300 and amyloid < 500  # CSF biomarker rule

# 4. Raman Spectroscopy
def raman_spectroscopy_test(data: Dict[str, Any]) -> bool:
    sig = data.get("raman_signature", 0.0)
    return sig > 0.8  # ML-derived threshold

# 5. Neuroimaging Using HRNet
def neuroimaging_test(data: Dict[str, Any]) -> bool:
    """
    Adapts HRNet-Facial-Landmark-Detection to classify AD vs. normal on MRI images.
    """
    repo_dir = "/tmp/HRNet-Facial-Landmark-Detection"
    if not os.path.isdir(repo_dir):
        git.Repo.clone_from(
            "https://github.com/HRNet/HRNet-Facial-Landmark-Detection.git",
            repo_dir
        )  # clone HRNet repo :contentReference[oaicite:0]{index=0}

    # Assume we have a modified model definition in local repo:
    from tools.mod_hrnet import get_hrnet_ad_model

    # Load the fine-tuned AD classifier
    model = get_hrnet_ad_model(pretrained=True)
    model.eval()  # freeze for inference :contentReference[oaicite:1]{index=1}

    # Preprocess MRI slice image
    img_path = data.get("mri_image_path")
    if not img_path or not os.path.exists(img_path):
        raise ValueError("Provide a valid 'mri_image_path'")
    img = Image.open(img_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    tensor = preprocess(img).unsqueeze(0)

    # Inference
    with torch.no_grad():
        logits = model(tensor)  # [1, 2]
        pred = logits.argmax(dim=1).item()
    return bool(pred)  # 1 -> AD, 0 -> non-AD :contentReference[oaicite:2]{index=2}

# Main Router
def select_and_predict(input_data: Dict[str, Any]) -> bool:
    """
    Dispatches to the appropriate diagnostic function.
    type in ['clinical','cognitive','lab','imaging','raman']
    """
    typ = input_data.get("type")
    if typ == "clinical":
        return clinical_evaluation(input_data)
    if typ == "cognitive":
        return cognitive_testing(input_data)
    if typ == "lab":
        return laboratory_test(input_data)
    if typ == "imaging":
        return neuroimaging_test(input_data)
    if typ == "raman":
        return raman_spectroscopy_test(input_data)
    raise ValueError(f"Unknown type: {typ}")

# Sample Inputs
if __name__ == "__main__":
    samples = [
        {"type": "clinical", "symptom_score": 7},
        {"type": "cognitive", "cognitive_score": 22},
        {"type": "lab", "amyloid_beta": 400, "tau": 350},
        {"type": "imaging", "mri_image_path": "brain_slice.png"},
        {"type": "raman", "raman_signature": 0.9},
    ]
    for s in samples:
        print(f"{s['type']}: AD={select_and_predict(s)}")
