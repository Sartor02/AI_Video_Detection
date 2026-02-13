# AI Video Detection

Project for detecting artificially generated videos using AI models.

## Prerequisites

* Python 3.11 or higher
* pip (Python package manager)
* CUDA (optional, but recommended for training and video generation)

## Installation

1. **Clone the repository** (or navigate to the project folder)

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
AI_Video_Detection/
├── datasets/                 # Generated datasets
│   ├── fake_pyramid/
│   ├── fake_sota/
│   ├── real/
│   └── real_generalization/
├── models/                   # Pre-trained models
├── results/                  # Experiment results
└── report.ipynb              # Notebook with analysis and results
```

## Usage

### 1. View Results

Open and run the main notebook:

```bash
jupyter notebook report.ipynb
```

## 🔬 Available Models

The project includes several trained models:

* **Baseline_DinoV2.pth** – Baseline with DINOv2
* **Baseline_Simple_ViT.pth** – Baseline with Simple ViT
* **incr_Baseline+Forensic.pth** – With forensic features
* **incr_Baseline+MultiScale.pth** – With multi-scale analysis
* **incr_Baseline+PyramidNoise.pth** – With pyramidal noise analysis
* **incr_Baseline+SpatialAttn.pth** – With spatial attention
* **incr_Baseline+TemporalLoss.pth** – With temporal loss
* **incr_Baseline+Wavelet.pth** – With wavelet analysis

## Results

Experiment results are available in the `results/` folder in JSON and CSV format.

## Troubleshooting

* **CUDA issues**: If you do not have a GPU, the code will still run on CPU (slower)
* **Insufficient memory**: Reduce the batch size in the training scripts
* **Missing dependencies**: Make sure all dependencies in `requirements.txt` are properly installed
