# ECG Sensor-Based Arrhythmia Classification System

A complete end-to-end system for real-time ECG acquisition, signal processing, and automatic arrhythmia classification. The pipeline spans custom analog hardware, an Arduino microcontroller, LabVIEW signal processing, and MATLAB-based machine learning models trained on the MIT-BIH Arrhythmia Database.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Hardware Design](#hardware-design)
- [Repository Structure](#repository-structure)
- [Software Requirements](#software-requirements)
- [Setup & Usage](#setup--usage)
  - [1. Hardware & Arduino Firmware](#1-hardware--arduino-firmware)
  - [2. LabVIEW Signal Processing](#2-labview-signal-processing)
  - [3. MATLAB Classification](#3-matlab-classification)
- [Dataset](#dataset)
- [Machine Learning Models](#machine-learning-models)
- [Documentation](#documentation)

---

## Overview

This project implements a real-time cardiac arrhythmia detection system using a custom-built 3-lead ECG sensor. The acquired signal travels through the following stages:

1. **Analog front-end** — 3-lead ECG sensor built around an instrumentation amplifier.
2. **Data acquisition** — Arduino samples the conditioned signal at ~125 Hz via a hardware timer and streams samples over UART.
3. **Signal processing** — LabVIEW receives the serial stream, reconstructs the waveform, computes an FFT to derive heart rate, and forwards the raw sample data to MATLAB over TCP/IP.
4. **Classification** — MATLAB receives the ECG segment, applies a Butterworth band-pass filter, and runs an arrhythmia classification model, returning the predicted class back to LabVIEW.

Five arrhythmia classes from the MIT-BIH dataset are supported:

| Label | Description |
|-------|-------------|
| 0 | Normal beat |
| 1 | Supraventricular ectopic beat |
| 2 | Ventricular ectopic beat |
| 3 | Fusion beat |
| 4 | Unknown / paced beat |

---

## System Architecture

```
┌──────────────────────┐     UART/Serial      ┌─────────────────────────┐
│  3-Lead ECG Sensor   │ ──────────────────►  │  Arduino (Timer DAQ)    │
│ (Instr. Amplifier)   │                       │  ~125 Hz, 10-bit ADC    │
└──────────────────────┘                       └────────────┬────────────┘
                                                            │ USB Serial (9600 baud)
                                                            ▼
                                               ┌─────────────────────────┐
                                               │  LabVIEW (FFT + WFM)    │
                                               │  • Waveform display     │
                                               │  • FFT / heart-rate     │
                                               └────────────┬────────────┘
                                                            │ TCP/IP (localhost)
                                                            ▼
                                               ┌─────────────────────────┐
                                               │  MATLAB Classification  │
                                               │  • Butterworth BPF      │
                                               │  • CNN or Random Forest │
                                               │  • Returns class label  │
                                               └─────────────────────────┘
```

---

## Hardware Design

- **Sensor topology:** 3-lead configuration (RA, LA, RL/ground).
- **Front-end:** Instrumentation amplifier providing high common-mode rejection ratio (CMRR) to isolate the millivolt-level cardiac signal from noise.
- **Microcontroller:** Arduino (Uno/Nano or compatible) samples the amplified output on analogue pin **A0**.
- **Sampling rate:** Hardware Timer 1 configured in CTC mode (prescaler 256, OCR1A = 500) yields ≈ **125 Hz** — matching the MIT-BIH recording rate.
- **Serial output:** 4-digit zero-padded ADC values transmitted at **9600 baud** over USB-Serial.

---

## Repository Structure

```
.
├── Arduino_Firmware/
│   └── TimerBasedDaq.ino        # Timer-based ADC acquisition & serial output
│
├── LABView_Processing/
│   └── FFT+wfm.vi               # LabVIEW VI: waveform display, FFT, TCP/IP bridge
│
├── MATLAB_Classification/
│   ├── ECG_Classification.m     # 1-D CNN model training (Adam, 5 epochs)
│   ├── ecgGradientBoot.m        # Random Forest (100-tree bagged ensemble) training
│   ├── ecgTest.m                # Standalone evaluation on MIT-BIH test set
│   ├── testingEcgScript.m       # CNN training with class-0 undersampling
│   ├── testtcp.m                # TCP client: receive → filter → classify → respond
│   └── testscript.m             # TCP server stub for local testing
│
└── Docs/
    ├── ECG-Project-Final-Report.pdf.pdf   # Detailed project report
    └── ECG-Project-Presentation.pdf.pdf  # Project presentation slides
```

> **Note:** Dataset files (`mitbih_train.csv`, `mitbih_test.csv`) and saved model files (`*.mat`) are excluded from version control via `.gitignore`. See the [Dataset](#dataset) section for download instructions.

---

## Software Requirements

| Tool | Version / Notes |
|------|-----------------|
| Arduino IDE | 1.8+ (or Arduino CLI) |
| LabVIEW | 2019+ with NI-VISA and NI-Serial drivers |
| MATLAB | R2020a+ with **Deep Learning Toolbox** and **Statistics and Machine Learning Toolbox** |

---

## Setup & Usage

### 1. Hardware & Arduino Firmware

1. Build the 3-lead ECG sensor circuit (refer to `Docs/ECG-Project-Final-Report.pdf.pdf` for schematic details).
2. Connect the sensor output to analogue pin **A0** of the Arduino.
3. Open `Arduino_Firmware/TimerBasedDaq.ino` in the Arduino IDE.
4. Select the correct board and COM port, then upload the sketch.
5. The Arduino will continuously stream 4-digit ADC samples over serial at 9600 baud.

### 2. LabVIEW Signal Processing

1. Open `LABView_Processing/FFT+wfm.vi` in LabVIEW.
2. Configure the serial resource name to match the Arduino COM port.
3. Run the VI. The front panel will display:
   - **Live ECG waveform** reconstructed from the incoming serial samples.
   - **FFT spectrum** with derived heart rate.
4. The VI also acts as a TCP/IP server/client to exchange data with MATLAB (default address `127.0.0.1`, port `5001`).

### 3. MATLAB Classification

#### Training a model

**CNN model (`ECG_Classification.m` or `testingEcgScript.m`):**
```matlab
% Place mitbih_train.csv in the MATLAB working directory, then run:
run('MATLAB_Classification/testingEcgScript.m')
% Saves: trainedECGModel1.mat
```

**Random Forest model (`ecgGradientBoot.m`):**
```matlab
run('MATLAB_Classification/ecgGradientBoot.m')
% Saves: trainedRandomForestModel.mat
```

#### Evaluating on the test set

```matlab
% Place mitbih_test.csv in the MATLAB working directory, then run:
run('MATLAB_Classification/ecgTest.m')
```

#### Real-time classification via TCP/IP

1. Ensure LabVIEW is running and streaming data on port `5001`.
2. In MATLAB, run `testtcp.m`:

```matlab
run('MATLAB_Classification/testtcp.m')
```

The script will:
- Connect to LabVIEW over TCP/IP.
- Receive a raw ECG segment.
- Apply a 4th-order Butterworth band-pass filter (0.5–10 Hz, fs = 125 Hz).
- Pad/reshape the segment to 187 samples.
- Classify using the loaded CNN model.
- Return the predicted arrhythmia class label to LabVIEW.

---

## Dataset

The models are trained and evaluated on the **MIT-BIH Arrhythmia Database** in pre-processed CSV format (187 features per beat + 1 label column), as used in the [ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) on Kaggle.

| File | Rows | Description |
|------|------|-------------|
| `mitbih_train.csv` | 87,554 | Training split |
| `mitbih_test.csv`  | 21,892 | Test split |

Download both files and place them in the `MATLAB_Classification/` directory (or your MATLAB working directory) before running any training or evaluation scripts.

---

## Machine Learning Models

### 1-D Convolutional Neural Network (CNN)

Implemented in `ECG_Classification.m` and `testingEcgScript.m`.

| Layer | Details |
|-------|---------|
| Input | `[187 × 1 × 1]` |
| Conv2D | 4 filters, kernel `[5×1]`, padding `same` |
| ReLU | — |
| MaxPool | `[2×1]`, stride `[2×1]` |
| Fully connected | 4 units + ReLU |
| Fully connected | 5 units (one per class) |
| Softmax + Classification | — |

- **Optimizer:** Adam
- **Epochs:** 5
- **Mini-batch size:** 32
- **Class imbalance handling:** Majority class (Normal) undersampled to 8,000 samples.

### Random Forest (Bagged Ensemble)

Implemented in `ecgGradientBoot.m`.

- **Algorithm:** Bagged decision trees (`fitcensemble` with `'Method','Bag'`)
- **Number of trees:** 100
- **Class imbalance handling:** Majority class undersampled before training.

---

## Documentation

Full design details, circuit schematics, experimental results, and analysis are available in the `Docs/` folder:

- **`ECG-Project-Final-Report.pdf.pdf`** — Complete project report covering system design, hardware implementation, software architecture, and classification results.
- **`ECG-Project-Presentation.pdf.pdf`** — Slide deck summarising the project for presentation purposes.
