# Brain Tumor Detection Audit

Date: 2026-05-25

## Environment

`rtx50_env` was found at `/home/nkpen/miniforge3/envs/rtx50_env`.

| Check | Result |
|---|---|
| Python | 3.11.14 |
| Torch | 2.9.0+cu128 |
| CUDA visible to Torch | Yes |
| CUDA runtime | 12.8 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| Gradio | 6.3.0 |
| MONAI | 1.5.1 |
| timm | 1.0.24 |
| Ultralytics | 8.4.5 |

`pip freeze` shows the environment is much larger than the project needs. Keep this env for experimentation, but use a fresh env from `requirements.txt` and `constraints.txt` for reproducible runs.

The full package inventory is stored in [`rtx50_env_freeze.txt`](rtx50_env_freeze.txt).

## Required Project Dependencies

The code imports these non-stdlib packages:

- `torch`, `torchvision`, `torchaudio`
- `ultralytics`
- `monai`, `einops`
- `timm`
- `opencv-python-headless`
- `Pillow`
- `nibabel`
- `pydicom`
- `gradio`
- `numpy`
- `scikit-learn`
- `pandas`
- `tqdm`
- `fpdf2`
- `requests`
- `kaggle`

These are covered by `requirements.txt` and pinned in `constraints.txt`.

## Findings

- `pip check` fails with `ValueError: too many values to unpack (expected 3)` inside pip's wheel tag parser. This prevents a clean dependency-health report.
- Both `fpdf==1.7.2` and `fpdf2==2.8.7` are installed. They share the `fpdf` module namespace; uninstall legacy `fpdf` and keep `fpdf2`.
- `opencv-python` and `opencv-python-headless` are both installed in `rtx50_env`; the project only needs the headless package.
- `black --check *.py` reports that all eight Python scripts would be reformatted.
- `mypy --ignore-missing-imports *.py` did not complete in a practical time window for this ML-heavy script set and was stopped.
- Docker is available and `docker compose config` validates. Rebuild verification found and fixed an invalid NVIDIA CUDA base-image tag and unpinned PyTorch install step.
- The rebuilt Compose service starts successfully on port `7860`, reports CUDA `12.8`, runs Torch `2.9.0+cu128`, sees the RTX 5060 GPU, and passes `pip check` inside the container.
- The Gradio theme API in Gradio 6.3 rejected raw string font lists in `gr.themes.Base(...)`; CSS system-font styling is used instead.
- Generated training/runtime data is intentionally kept out of git: `dataset_*`, `tmp_*`, model weights, `runs/`, `reports/`, `test_artifacts/`, generated PDFs, and generated ODF/Office documents.

## Verification

Commands run:

```bash
/home/nkpen/miniforge3/envs/rtx50_env/bin/python -m compileall -q *.py
/home/nkpen/miniforge3/envs/rtx50_env/bin/python -m py_compile 06_clinical_diagnostic_interface.py
BTD_SKIP_MODEL_LOAD=1 /home/nkpen/miniforge3/envs/rtx50_env/bin/python -m unittest discover -s tests -v
docker compose config
docker compose build btd-ai
docker compose up -d btd-ai
docker compose exec -T btd-ai python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
docker compose exec -T btd-ai python -m pip check
curl -I --max-time 5 http://127.0.0.1:7860
```

Browser verification:

- Installed Playwright Chromium because browser automation was missing.
- Launched the real Gradio app at `http://127.0.0.1:7860`, including a final pass against the Docker-served app.
- Confirmed the dashboard rendered.
- Uploaded `dataset_ensemble/Glioma/mri_Te-glTr_0000.jpg`.
- Clicked **Analyse scan**.
- Verified that a diagnostic result rendered.
- Captured screenshots and video under `docs/media/`.

Browser console messages:

- Two warnings: `Method not implemented.`
- No page errors were observed during the e2e run.

## Documentation Sources

- Gradio Themes: https://www.gradio.app/docs/gradio/themes
- Gradio Blocks `theme` and `css` parameters: https://www.gradio.app/docs/gradio/blocks
- Apple Human Interface Guidelines: https://developer.apple.com/design/human-interface-guidelines/
