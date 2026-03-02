# AI Video Detection
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/Sartor02/AI_Video_Detection)

This repository contains a project for detecting artificially generated videos using a 3D Vision Transformer (3D-ViT) architecture. The study systematically evaluates various augmentation strategies and architectural components to improve a model's ability to generalize and detect videos from state-of-the-art (SOTA) generators like Sora and PyramidFlow.

The core of the analysis is presented in `report.ipynb`, which details an ablation study to measure the impact of techniques such as forensic augmentation, wavelet-based blending, multi-scale temporal sampling, and spatial attention.

## Project Structure

```
sartor02-ai_video_detection/
├── datasets/                 # Datasets for training and testing
│   ├── fake_pyramid/         # Fake videos from PyramidFlow (for training)
│   ├── fake_sota/            # Fake videos from SOTA generators (for testing)
│   ├── real/                 # Real videos (for training)
│   └── real_generalization/  # Unseen real videos (for testing)
├── results/                  # JSON and CSV files with experiment metrics
├── report.ipynb              # Jupyter notebook with all code, analysis, and visualizations
└── requirements.txt          # Python package dependencies
```

## Prerequisites

*   Python 3.11 or higher
*   pip (Python package manager)
*   CUDA (optional, but highly recommended for training)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sartor02/ai_video_detection.git
    cd ai_video_detection
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # Create the environment
    python -m venv venv

    # Activate on Windows
    .\venv\Scripts\activate

    # Activate on macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The entire project, from data loading and model training to final analysis, is contained within the `report.ipynb` Jupyter notebook.

To view the experiments and results, start the Jupyter environment:

```bash
jupyter notebook report.ipynb
```

Inside the notebook, you can execute the cells to reproduce the training process, generate result plots, and explore the detailed analysis of model performance. Pre-trained models and saved results are included in the repository, so training is not required to view the final outcomes.

## Ablation Study and Models

The project evaluates several components by incrementally adding them to a baseline ViT model. The effectiveness of each component is measured by its impact on generalization to unseen SOTA-generated videos.

The trained models include:

*   **Baseline\_Simple\_ViT**: A standard Vision Transformer (ViT-Base) without special augmentations.
*   **Baseline\_DinoV2**: A baseline using the powerful DINOv2 backbone.
*   **incr\_Baseline+Forensic**: Adds a forensic augmentation pipeline (JPEG, blur, noise, etc.) to improve robustness.
*   **incr\_Baseline+MultiScale**: Implements multi-scale temporal sampling (4, 8, or 16 frames) to improve temporal robustness.
*   **incr\_Baseline+PyramidNoise**: Adds Gaussian noise to simulate diffusion model artifacts.
*   **incr\_Baseline+SpatialAttn**: Incorporates a spatial attention module to help the model focus on artifact-rich regions.
*   **incr\_Baseline+TemporalLoss**: Adds an auxiliary temporal consistency loss using an LSTM.
*   **incr\_Baseline+Wavelet**: Uses wavelet blending to create "clean fakes," forcing the model to learn semantic and temporal anomalies instead of frequency artifacts.

## Key Results

The study's primary goal was to find an architecture that generalizes well to challenging, unseen fakes.

| Model Configuration | Validation Accuracy | SOTA Test Accuracy | Generalization Gap |
| :------------------ | :-----------------: | :----------------: | :----------------: |
| Baseline (Simple ViT) |       95.00%        |       96.30%       |       -1.30%       |
| **MultiScale**      |     **97.50%**      |    **97.53%**    |     **-0.03%**     |
| Forensic            |       97.50%        |       95.06%       |       +2.44%       |
| Wavelet             |       96.67%        |       93.83%       |       +2.84%       |
| Temporal Loss       |       96.67%        |       93.83%       |       +2.84%       |

### Conclusion

*   The **Multi-Scale Temporal Sampling** approach demonstrated the best performance, achieving **97.53% accuracy** on the SOTA test set and effectively closing the generalization gap to nearly zero. This highlights that temporal robustness is a key factor in detecting modern AI-generated videos.
*   Wavelet and FFT analysis revealed that the "hardest" fakes to detect have frequency characteristics that are much closer to those of real videos, explaining why models often fail to identify them.

## Troubleshooting

*   **CUDA Issues**: If you do not have a compatible GPU, the code will automatically fall back to using the CPU, which will be significantly slower.
*   **Memory Errors**: If you encounter `OutOfMemoryError` during training, try reducing the `BATCH_SIZE` hyperparameter in the notebook.
*   **Missing Dependencies**: Ensure all packages listed in `requirements.txt` are installed correctly within your active virtual environment.
