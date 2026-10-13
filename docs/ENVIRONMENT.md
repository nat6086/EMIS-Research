
# EMIS-Research — Computational Environment

## Operating system
- Windows
- IDE: Visual Studio Code
- Python: 3.12.10
- Environment: .venv

## GPU
- NVIDIA GeForce RTX 4060 Laptop GPU
- VRAM: approximately 8 GB
- NVIDIA driver: 592.82
- Maximum CUDA version reported by nvidia-smi: 13.1

## Core dependencies
- PyTorch: 2.11.0+cu130
- torchaudio: 2.11.0+cu130
- NumPy: 1.26.4

## PyTorch installation

The CUDA-enabled PyTorch packages were installed using:

```powershell
python -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
```

The installation required an increased network timeout due to slow downloads.

## Validation

The following checks were successful:

- PyTorch imported successfully.
- CUDA was available.
- The NVIDIA RTX 4060 was detected.
- A tensor operation executed successfully on the GPU.
- torchaudio imported successfully.
- PyTorch-to-NumPy conversion was successful.

## Dependency snapshot

Exact installed Python dependencies are recorded in:

`requirements-lock.txt`

This snapshot is specific to the validated Windows environment.
