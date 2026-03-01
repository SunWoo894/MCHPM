# MCHPM

Official implementation of:
> Lim, H., Park, S., Li, Q., Li, X., & Kim, J. (2026).
**What makes a review helpful? A multimodal prediction model in e-commerce**. 
_Electronic Commerce Research and Applications_, 76, 101586. [Paper](https://doi.org/10.1016/j.elerap.2026.101586)

## Overview
This repository provides the official implementation of MCHPM (Multimodal Cue-based Helpfulness Prediction Model), a theory-driven deep learning framework for review helpfulness prediction in e-commerce. MCHPM is grounded in the Elaboration Likelihood Model and reflects how consumers evaluate online reviews through central and peripheral information-processing routes.

Existing MRHP (Multimodal Review Helpfulness Prediction) models primarily focus on deep semantic representations from text and images while overlooking shallow cues such as readability and image quality. To address this limitation, MCHPM systematically integrates central cues extracted via BERT and VGG-16 with peripheral cues computed from textual and visual surface features.

A co-attention mechanism models the interdependencies between central and peripheral cues within each modality, and a Gated Multimodal Unit dynamically adjusts the relative importance of text and image representations during prediction. Experiments on large-scale Amazon datasets demonstrate that MCHPM consistently outperforms strong unimodal and multimodal baselines, achieving average improvements of 3.864% in MAE, 4.061% in MSE, 2.172% in RMSE, and 6.349% in MAPE. These results validate the effectiveness of theory-driven multimodal cue integration for review helpfulness prediction.

## Requirements
- python >= 3.9
- torch == 2.3.1
- torchvision == 0.18.1
- tensorflow == 2.15.0
- transformers == 4.28.1
- tokenizers == 0.13.3
- sentencepiece == 0.2.0
- huggingface-hub == 0.23.4
- nltk == 3.9.2
- textblob == 0.19.0
- textstat == 0.7.11
- numpy == 1.26.4
- pandas == 2.2.1
- pyarrow == 12.0.1
- scikit-learn == 1.4.2
- opencv-python
- Pillow == 10.3.0
- tqdm == 4.66.4
- PyYAML == 6.0.1

## Repository Structure
Below is the project structure for quick reference.

```bash
├── data/                        # Dataset directory
│   ├── raw/                     # Original (unprocessed) datasets
│   └── processed/               # Preprocessed data for training and evaluation
│
├── model/                       # MCHPM architecture and training pipeline
│   └── proposed.py              # End-to-end MCHPM implementation
│
├── src/                         # Core source code
│   ├── data.py                  # Data preprocessing and dataset loader
│   ├── bert.py                  # Text central cue extraction using BERT
│   ├── vgg16.py                 # Image central cue extraction using VGG-16
│   ├── peripheral_features.py   # Peripheral cue extraction pipeline for text and images
│   ├── image_manager.py         # Image downloading and path management utilities
│   ├── config.yaml              # Model and training configuration file
│   ├── path.py                  # Path and directory management utilities
│   └── utils.py                 # Helper functions (metrics and logging)
│
├── main.py                      # Entry point for model training and evaluation
│
├── requirements.txt             # Python package dependencies
│
├── README.md                    # Project documentation
│
└── .gitignore                   # Git ignore configuration
```

## Model Description

MCHPM (Multimodal Cue-based Helpfulness Prediction Model) is a theory-driven review helpfulness prediction framework designed to reflect consumers’ dual-route information processing mechanism. Grounded in the Elaboration Likelihood Model, MCHPM explicitly models both central cues (deep semantic and visual representations) and peripheral cues (surface-level textual and image-quality features) within a unified multimodal architecture.

The model consists of three main modules:
- **Multi-Cue Extraction Module:** Extracts central and peripheral cues from review text and images.
- **Cue-Integration Module:** Models the interdependencies between central and peripheral cues within each modality.
- **Multimodal Fusion Module:** Dynamically fuses textual and visual representations to predict review helpfulness.

In the Multi-Cue Extraction module, textual central features are obtained from BERT, while visual central features are extracted from VGG-16. Peripheral cues, including sentiment, subjectivity, readability, extremity, brightness, contrast, saturation, and edge intensity, are computed using Python-based feature extraction. These cues represent shallow attributes that influence consumers’ evaluation processes.

In the Cue-Integration module, a co-attention mechanism captures the interactions between textual and visual representations. This mechanism enables the model to learn how features from one modality inform and refine the representations of the other. Feed-forward layers and residual connections further stabilize and enhance feature learning.

In the Multimodal Fusion module, a GMU (Gated Multimodal Fusion) mechanism dynamically adjusts the relative importance of text and image modalities. The fused representation is then passed to a multilayer perceptron for final helpfulness score prediction.

<p align="center">
  <img src="data/MCHPM Architecture.png" alt="MCHPM Architecture" width="800">
</p>

## How to Run

### Environment Setup
Create a virtual environment (Python ≥ 3.9 recommended) and install the required dependencies:

#### Option A: Using venv
```bash
python3.9 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

#### Option B: Using Conda
```bash
conda create -n mchpm python=3.9
conda activate mchpm
pip install -r requirements.txt
```

### Data Preparation
Place your dataset under `data/raw/` and ensure that its format matches the preprocessing pipeline defined in `src/data.py`.

Preprocessed data will be stored under `data/processed/` after feature extraction.

### Configuration
Edit `src/config.yaml` to configure training, data paths, and model hyperparameters before running the experiment.

### Train and Evaluate the Model
Run the training and evaluation script:
```bash
python main.py
```

## Experimental Results

MCHPM was evaluated on two large-scale Amazon review datasets: Cell Phones & Accessories and Electronics.
The results demonstrate that MCHPM consistently outperforms strong unimodal and multimodal baselines across all evaluation metrics, achieving average improvements of 3.864% in MAE, 4.061% in MSE, 2.172% in RMSE, and 6.349% in MAPE compared with the strongest benchmark model.

<div align="center">
  <table> 
    <thead> 
      <tr>
        <th rowspan="2">Model</th>
        <th colspan="4">Cell Phones & Accessories</th> 
        <th colspan="4">Electronics</th> 
      </tr>
      <tr> 
        <th>MAE</th> 
        <th>MSE</th> 
        <th>RMSE</th> 
        <th>MAPE</th>
        <th>MAE</th>
        <th>MSE</th>
        <th>RMSE</th>
        <th>MAPE</th>
      </tr>
    </thead> 
    <tbody> 
      <tr> 
        <td>LSTM</td> 
        <td>0.647</td><td>0.821</td><td>0.849</td><td>56.702</td> 
        <td>0.711</td><td>0.896</td><td>0.946</td><td>57.678</td> 
      </tr>
      <tr>
        <td>TNN</td>
        <td>0.643</td><td>0.714</td><td>0.845</td><td>56.650</td>
        <td>0.722</td><td>0.904</td><td>0.851</td><td>59.556</td>
      </tr>
      <tr>
        <td>DMAF</td>
        <td>0.625</td><td>0.691</td><td>0.836</td><td>53.139</td>
        <td>0.697</td><td>0.880</td><td>0.939</td><td>55.198</td>
      </tr>
      <tr>
        <td>CS-IMD</td>
        <td>0.615</td><td>0.681</td><td>0.825</td><td>52.392</td>
        <td>0.687</td><td>0.831</td><td>0.912</td><td>56.032</td>
      </tr>
      <tr>
        <td><b>MFRHP (Proposed)</b></td>
        <td><b>0.625</b></td><td><b>0.695</b></td><td><b>0.837</b></td><td><b>53.116</b></td>
        <td><b>0.695</b></td><td><b>0.840</b></td><td><b>0.916</b></td><td><b>57.488</b></td>
      </tr>
    </tbody>
  </table>
</div>
      
## Citation

If you use this repository in your research, please cite:

```bibtex
@article{LIM2026101586,
  title = {What makes a review helpful? A multimodal prediction model in e-commerce},
  author = {Heena Lim and Seonu Park and Qinglong Li and Xinzhe Li and Jaekyeong Kim},
  journal = {Electronic Commerce Research and Applications},
  volume = {76},
  pages = {101586},
  year = {2026},
  doi = {10.1016/j.elerap.2026.101586}  
}
```

## Contact

For research inquiries or collaborations, please contact:  

**Seonu Park**  
Ph.D. Student, Department of Big Data Analytics  
Kyung Hee University  
Email: sunu0087@khu.ac.kr

**Qinglong Li**  
Assistant Professor, Division of Computer Engineering  
Hansung University  
Email: leecy@hansung.ac.kr

_Last updated: March 2026_
