# MDM — Human Motion Diffusion Model

Text-to-motion generation model used as the motion backend in the [NTHU Avatar pipeline](https://github.com/sirawee-lee/ml-hugs-NTHUavatar2).

Based on [Human Motion Diffusion Model](https://arxiv.org/abs/2209.14916) (ICLR 2023) by Guy Tevet et al.

---

## Setup

### 1. Create environment

```bash
conda env create -f environment.yml
conda activate mdm
pip install -e .
```

### 2. Download required assets

```bash
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
```

### 3. Download pretrained model

Download the **50-step HumanML3D model** (~1.5 GB) from [Google Drive](https://drive.google.com/file/d/1cfadR1eZ116TIdXK7qDX1RugAerEiJXr/view) and place it at:

```
save/humanml_enc_512_50steps/model000750000.pt
```

### 4. Download HumanML3D dataset

Follow the instructions in `prepare/` to download the HumanML3D text-motion dataset into `dataset/`.

---

## Usage

### Standalone motion generation

```bash
conda activate mdm
python -m sample.generate \
  --model_path save/humanml_enc_512_50steps/model000750000.pt \
  --text_prompt "a person waves hello" \
  --num_samples 4 \
  --guidance_param 2.5
```

Output is saved to `save/humanml_enc_512_50steps/samples_<timestamp>/`.

### In the NTHU Avatar pipeline

This repo is called automatically by `run_text2hugs.py` in the HUGS repo. No manual steps needed — just make sure the environment and checkpoints are in place.

The two scripts used by the pipeline are:
- `sample/generate.py` — generates motion from text prompt
- `sample/extract_smpl_params.py` — converts MDM output to HUGS-compatible SMPL `.npz`

---

## Project Structure

```
motion-diffusion-model/
├── sample/
│   ├── generate.py              # Main generation script
│   └── extract_smpl_params.py   # MDM → HUGS SMPL converter
├── model/                       # Diffusion model architecture
├── diffusion/                   # Diffusion process
├── data_loaders/                # HumanML3D data loading
├── train/                       # Training scripts
├── eval/                        # Evaluation scripts
├── prepare/                     # Asset download scripts
├── save/                        # Model checkpoints (gitignored)
├── body_models/                 # SMPL body models (gitignored)
├── dataset/                     # HumanML3D dataset (gitignored)
└── environment.yml
```

---

## Citation

```bibtex
@inproceedings{tevet2023human,
  title={Human Motion Diffusion Model},
  author={Guy Tevet and Sigal Raab and Brian Gordon and Yoni Shafir and Daniel Cohen-or and Amit Haim Bermano},
  booktitle={ICLR},
  year={2023},
  url={https://openreview.net/forum?id=SJ1kSyO2jwu}
}
```
