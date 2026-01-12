# MCHPM

Official implementation of the paper:
> **Lim, H., Park, S., Li, Q., Li, X., & Kim, J. (2026). What Makes a Review Helpful? A Multimodal Prediction Model in E-Commerce. Electronic Commerce Research and Applications,**  [Paper Link](https://doi.org/10.1016/j.ins.2026.123078) -> 수정예정
>
> ## Overview
This repository provides the official implementation of MCHPM (Multimodal Cue-based Helpfulness Prediction Model), a deep learning framework for predicting the helpfulness of online reviews in e-commerce platforms. Grounded in the Elaboration Likelihood Model, MCHPM distinguishes between central cues and peripheral cues in multimodal reviews. Central cues are extracted from textual and visual content using BERT and VGG-16 to capture rich semantic information, while peripheral cues such as text readability and image quality are derived through Python-based feature extraction. A co-attention mechanism is employed to model interdependencies among multimodal cues, and a Gated Multimodal Unit adaptively weights the contribution of each modality during prediction. Experiments conducted on a real-world Amazon review dataset demonstrate that MCHPM achieves average improvements of 3.864% in MAE, 4.061% in MSE, 2.172% in RMSE, and 6.349% in MAPE over the strongest competing methods, highlighting the importance of incorporating shallow features into multimodal review helpfulness prediction.

## Requirements
- tensorflow==2.15.0
- torchvision==0.18.1
- torch==2.3.1
- transformers==4.28.1
﻿- huggingface-hub==0.23.4
- scikit-learn==1.4.2
- numpy==1.26.4
- pandas==2.2.1
- pyarrow==12.0.1
- PyYAML==6.0.1
- sentencepiece==0.2.0
- textblob==0.19.0
- textstat==0.7.11
- tokenizers==0.13.3
- tqdm==4.66.4
- Pillow==10.3.0
- nltk==3.9.2
- opencv-python

## Repository Structure
Below is the project structure for quick reference.


```bash
├── data/                        # Dataset directory
│   ├── raw/                     # Original (unprocessed) datasets
│   └── processed/               # Preprocessed data for training/evaluation
│
├── model/                       # Model definitions and checkpoints
│
├── src/                         # Core source code
│   ├── data.py                  # data preprocessing module
│   ├── bert.py                  # Text central cues (BERT-based embedding) extraction module
│   ├── vgg16.py                 # Image central cues (BERT-based embedding) extraction module
│   ├── peripheral_features.py   # Peripheral cues (textblob, textstat, opencv) extraction module
│   ├── config.yaml              # Model and training configuration file
│   ├── path.py                  # Path and directory management utilities
│   └── utils.py                 # Helper functions (data loading, metrics, etc.)
│
├── main.py                      # Entry point for model training and evaluation
│
├── requirements.txt             # Python package dependencies
│
├── README.md                    # Project documentation
│
└── .gitignore                   # Git ignore configuration

```

