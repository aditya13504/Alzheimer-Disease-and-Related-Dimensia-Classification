import pytest
from router import select_and_predict

def test_clinical():
    assert select_and_predict({"type": "clinical", "symptom_score": 7}) is True
    assert select_and_predict({"type": "clinical", "symptom_score": 3}) is False

def test_cognitive():
    assert select_and_predict({"type": "cognitive", "test_score": 10}) is True
    assert select_and_predict({"type": "cognitive", "test_score": 20}) is False

def test_lab():
    assert select_and_predict({"type": "lab", "biomarker_level": 3.5}) is True
    assert select_and_predict({"type": "lab", "biomarker_level": 2.0}) is False

def test_raman():
    assert select_and_predict({"type": "raman", "spectrum_path": "some/path.csv"}) is True
    assert select_and_predict({"type": "raman"}) is False

def test_imaging(monkeypatch):
    # Patch imaging_ad_prediction to avoid actual model loading
    from diagnostics import imaging
    monkeypatch.setattr(imaging, "imaging_ad_prediction", lambda data: data.get("image_path") == "positive.png")
    assert select_and_predict({"type": "imaging", "image_path": "positive.png"}) is True
    assert select_and_predict({"type": "imaging", "image_path": "negative.png"}) is False
