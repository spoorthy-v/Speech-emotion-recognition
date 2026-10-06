# Speech-emotion-recognition# Speech Emotion Recognition: A Reproducible Benchmark Framework

A research-oriented benchmark framework for **Speech Emotion Recognition (SER)** using conventional acoustic representations and pretrained speech representations.

The project investigates how different speech representations and deep learning architectures perform for emotion classification under **speaker-independent evaluation**.

---

## Research Objective

Speech Emotion Recognition aims to automatically identify the emotional state conveyed in spoken language.

The central research question of this project is:

> **How do conventional acoustic representations compare with pretrained self-supervised speech representations for speaker-independent speech emotion recognition?**

The project is designed as a reproducible benchmark rather than a single-model implementation. Models, datasets, preprocessing, evaluation protocols, and experiments are organized to support systematic comparison.

---

## Current Status

### Completed

* RAVDESS dataset exploration
* RAVDESS metadata generation
* Speaker-independent train/validation/test split
* Audio loading and preprocessing pipeline
* Mel-spectrogram feature extraction
* PyTorch Dataset implementation
* PyTorch DataLoader implementation
* CNN baseline architecture
* Initial training pipeline
* Exploratory data analysis notebook
* Reproducible project structure

### In Progress

* Baseline model evaluation
* Model checkpointing
* Macro-F1 and weighted-F1 evaluation
* Per-class performance analysis
* Confusion matrix analysis
* Training-history visualization

### Planned

* CRNN / CNN-BiLSTM models
* Wav2Vec2-based SER
* HuBERT-based SER
* Whisper encoder-based representations
* Cross-dataset evaluation
* Explainability analysis
* Representation-level comparison
* Novel research contribution based on experimental findings

---

## Dataset

The initial experiments use the **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**.

The speech portion contains recordings representing eight emotion classes:

| Emotion   | Label |
| --------- | ----: |
| Neutral   |     1 |
| Calm      |     2 |
| Happy     |     3 |
| Sad       |     4 |
| Angry     |     5 |
| Fearful   |     6 |
| Disgust   |     7 |
| Surprised |     8 |

The dataset contains recordings from **24 actors**.

The audio is resampled to **16 kHz** and converted to mono during loading.

> The dataset itself is not included in this repository. It must be obtained separately from the official dataset source.

---

## Experimental Protocol

A **speaker-independent** evaluation protocol is used to prevent recordings from the same speaker appearing in multiple splits.

The 24 actors are divided into:

| Split      | Actors | Recordings |
| ---------- | -----: | ---------: |
| Training   |     16 |        960 |
| Validation |      4 |        240 |
| Test       |      4 |        240 |
| **Total**  | **24** |   **1440** |

The actor split is generated using a fixed random seed:

```text
Random seed: 42
```

The test speakers are not used during model training or model selection.

---

## Baseline Feature Representation

The current baseline uses **log-Mel spectrograms**.

Configuration:

| Parameter      |     Value |
| -------------- | --------: |
| Sample rate    |    16 kHz |
| Audio duration | 3 seconds |
| FFT size       |      1024 |
| Hop length     |       256 |
| Mel bands      |       128 |
| Channels       |         1 |

The resulting input to the CNN has the form:

```text
(batch_size, 1, 128, time_frames)
```

For the current 3-second configuration, the observed input shape is approximately:

```text
(16, 1, 128, 188)
```

---

## CNN Baseline

The current baseline CNN consists of three convolutional blocks followed by adaptive average pooling and a fully connected classifier.

Architecture:

```text
Input
  │
  ├── Conv2D (1 → 32)
  ├── BatchNorm
  ├── ReLU
  ├── MaxPool
  │
  ├── Conv2D (32 → 64)
  ├── BatchNorm
  ├── ReLU
  ├── MaxPool
  │
  ├── Conv2D (64 → 128)
  ├── BatchNorm
  ├── ReLU
  │
  ├── Adaptive Average Pooling
  │
  └── Linear Layer
          ↓
      8 emotion classes
```

The model is implemented in:

```text
src/models/cnn.py
```

---

## Repository Structure

```text
speech-emotion-recognition/
│
├── data/
│   ├── raw/
│   │   └── RAVDESS/              # Not tracked by Git
│   ├── processed/                # Not tracked by Git
│   └── metadata/                 # Generated metadata, not tracked
│
├── notebooks/
│   └── 01_ravdess_exploration.ipynb
│
├── src/
│   ├── datasets/
│   │   ├── dataloader.py
│   │   ├── ravdess.py
│   │   ├── ser_dataset.py
│   │   └── split_ravdess.py
│   │
│   ├── features/
│   │   └── mel_spectrogram.py
│   │
│   ├── models/
│   │   └── cnn.py
│   │
│   └── training/
│       └── train.py
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Environment Setup

Create and activate a Python environment, then install the required dependencies.

```bash
pip install -r requirements.txt
```

The current development environment uses Python 3.11.

---

## Running the Pipeline

### 1. Generate RAVDESS metadata

From the repository root:

```bash
python -m src.datasets.ravdess
```

### 2. Create the speaker-independent split

```bash
python -m src.datasets.split_ravdess
```

### 3. Train the CNN baseline

```bash
python -m src.training.train
```

The project uses module-based execution (`python -m`) so that imports from the `src` package work correctly from the repository root.

---

## Reproducibility

The project follows a reproducibility-oriented workflow:

* Fixed random seed for dataset splitting
* Speaker-independent evaluation
* Explicit preprocessing configuration
* Separate training, validation, and test speakers
* Version-controlled source code
* Dataset files excluded from Git
* Generated files separated from source code
* Experiment configurations documented in the repository

Future experiments will additionally record model configurations, evaluation metrics, and trained checkpoints.

---

## Evaluation

The final benchmark will report more than accuracy.

Planned metrics include:

* Accuracy
* Macro-F1
* Weighted-F1
* Per-class precision
* Per-class recall
* Per-class F1-score
* Confusion matrix

**Macro-F1** will be emphasized because it gives equal importance to each emotion class and is therefore more informative than accuracy alone when class distributions differ.

---

## Research Roadmap

```text
RAVDESS
   │
   ▼
Speaker-Independent Split
   │
   ▼
Log-Mel Spectrogram
   │
   ▼
CNN Baseline
   │
   ├──────────────► CRNN / CNN-BiLSTM
   │
   ├──────────────► Wav2Vec2
   │
   ├──────────────► HuBERT
   │
   └──────────────► Whisper Encoder
                         │
                         ▼
                 Representation Comparison
                         │
                         ▼
                  Cross-Dataset Evaluation
                         │
                         ▼
                    Explainability
                         │
                         ▼
                 Research Contribution
```

---

## Experimental Results

The initial experiments use the RAVDESS dataset with a speaker-independent
16/4/4 actor split and the same Log-Mel preprocessing configuration.

| Experiment | Representation | Model | Test Accuracy | Test Macro-F1 | Test Weighted-F1 |
|------------|----------------|-------|---------------|---------------|------------------|
| 1 | Log-Mel Spectrogram | CNN | 44.17% | 41.28% | 41.64% |
| 2 | Log-Mel Spectrogram | CNN-BiLSTM | **51.67%** | **51.09%** | **51.48%** |

### Experiment 2: CNN-BiLSTM

The CNN-BiLSTM extends the CNN baseline with bidirectional temporal
modeling. The CNN extracts local time-frequency features from the
Log-Mel spectrogram, while the BiLSTM models temporal dependencies
across the extracted feature sequence.

The best checkpoint was selected using validation Macro-F1.

- Best checkpoint: Epoch 22
- Validation Macro-F1: 50.67%
- Test Accuracy: 51.67%
- Test Macro-F1: 51.09%
- Test Weighted-F1: 51.48%

Compared with the CNN baseline, the CNN-BiLSTM improved test Macro-F1
from 41.28% to 51.09%, an absolute improvement of 9.81 percentage
points.

The same dataset split, preprocessing pipeline, and evaluation protocol
were retained to make the comparison between the two architectures
controlled and reproducible.


## Future Datasets

Following the initial RAVDESS benchmark, the framework is intended to support additional datasets such as:

* CREMA-D
* IEMOCAP
* MELD
* Other multilingual or Indian-language emotional speech datasets

Cross-dataset experiments will be used to investigate **domain and speaker generalization**.

---

## Research Direction

The longer-term objective is to move beyond conventional SER classification toward robust and transferable speech representations.

Potential research directions include:

* Self-supervised speech representation analysis
* Cross-corpus emotion recognition
* Low-resource and multilingual SER
* Indian-language speech emotion recognition
* Speaker-independent emotion recognition
* Explainable speech emotion recognition
* Robustness to recording conditions and domains

---



## License

This repository contains research code. Dataset licensing and usage conditions of external datasets remain the responsibility of the user.

See the repository `LICENSE` file for the code license.
