# Alzheimer Disease and Related Dementia (ADRD) Classification using HRNet

This repository provides a comprehensive framework for multi-modal Alzheimer's Disease and Related Dementia (ADRD) classification, leveraging High-Resolution Networks (HRNets) for neuroimaging and integrating clinical, cognitive, laboratory, and Raman spectroscopy data.

---

## Project Overview
- **Multi-modal ADRD diagnosis:** Integrates clinical, cognitive, lab, Raman, and imaging data.
- **HRNet-based neuroimaging:** Uses HRNet for MRI-based AD classification.
- **Two-stage AD confirmation:** Optionally confirms AD with plasma p-tau217 blood test.
- **Configurable and extensible:** YAML-based experiment configs, modular codebase.
- **Pretrained models and reproducible results.**

---

## Directory Structure
- `main.py`: Main entry point for multi-modal ADRD diagnosis.
- `router.py`, `ago.py`: Central routers for dispatching input to diagnostic modules.
- `diagnostics/`: Contains modules for each diagnostic technique (clinical, cognitive, lab, raman, imaging).
- `lib/`: Core library with model definitions, datasets, configs, and utilities.
- `tools/`: Training and testing scripts for HRNet models.
- `experiments/`: YAML config files for different datasets and experiments.
- `train/`, `test/`: Folders for training and testing data.
- `output/`, `log/`, `checkpoints/`: Output directories for logs, model checkpoints, and results.

---

## Quick Start

### Environment
- Python 3.6+
- PyTorch 1.0+
- Recommended: NVIDIA GPU with CUDA support

### Installation
1. Install PyTorch following the [official instructions](https://pytorch.org/)
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Clone the project:
   ```powershell
   git clone https://github.com/aditya13504/Alzheimer-Disease-and-Related-Dimensia-Classification.git
   cd Alzheimer-Disease-and-Related-Dimensia-Classification
   ```


### Data Preparation
- Download and organize your data as described in the experiment YAML files under `experiments/`.
- Example directory structure:
  ```
  Alzheimer-Disease-and-Related-Dimensia-Classification
  ├── lib
  ├── experiments
  ├── tools
  ├── data
  │   ├── 300w
  │   ├── aflw
  │   ├── cofw
  │   ├── wflw
  │   └── ...
  ├── train
  ├── test
  └── ...
  ```

---

## Training
Specify the configuration file in `experiments/`:
```powershell
python tools/train.py --cfg experiments/alzheimer/face_alignment_ad_hrnet_w18.yaml
```

## Testing
```powershell
python tools/test.py --cfg experiments/alzheimer/face_alignment_ad_hrnet_w18.yaml --model-file <MODEL WEIGHT>
```

## Multi-Modal Diagnosis (Main Entry Point)
Run the main router for ADRD diagnosis:
```powershell
python main.py --type <technique> [--biomarker_level <float>] [--image_path <path>] [--ptau217_level <float>]
```
- `--type`: One of `clinical`, `cognitive`, `lab`, `raman`, `imaging`
- `--ptau217_level`: (Optional) Plasma p-tau217 level for two-stage confirmation

### Two-Stage AD Confirmation
If any technique detects AD (even very mild), and you provide `--ptau217_level`, the system will run the plasma p-tau217 blood test for maximum accuracy. AD is only confirmed if both the initial technique and the blood test are positive.
- `--ptau217_level` (float, pg/mL): Plasma phosphorylated tau 217 level. Threshold: >0.37 pg/mL is highly predictive of AD.

#### Example Commands
- **Lab:**
  ```powershell
  python main.py --type lab --biomarker_level 3.2 --ptau217_level 0.45
  ```
- **Imaging with blood test confirmation:**
  ```powershell
  python main.py --type imaging --image_path path/to/mri.png --ptau217_level 0.45
  ```

---

## Acknowledgements
This project builds upon the official HRNet codebase and extends it for ADRD multi-modal diagnosis.

## License
MIT

## Contributions
Always open to contributions