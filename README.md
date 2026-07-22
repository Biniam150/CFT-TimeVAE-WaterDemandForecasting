# Conditional Forecasting TimeVAE (CFT-VAE) for Water Demand Forecasting

This repository provides a conditional TimeVAE-based framework for generating synthetic water-demand sequences and evaluating their usefulness for demand forecasting across multiple user-aggregation levels. It includes data preprocessing, conditional feature engineering, CFT-VAE and TimeVAE training, synthetic data generation, forecasting experiments, and statistical validation.

## 1. Overview

The Conditional Forecasting TimeVAE (CFT-VAE) framework is designed for:

- Synthetic water-demand data generation
- Data augmentation for water-demand forecasting
- Forecasting at different user-aggregation levels
- Comparing CFT-VAE with the original TimeVAE
- Evaluating real-only, synthetic-only, and augmented training scenarios
- Assessing synthetic-data fidelity, robustness, and statistical significance

The experimental pipeline supports 25 user-aggregation levels and multiple randomized trials. Forecasting performance is evaluated using six models:
- Random Forest
- XGBoost
- Gradient Boosting
- Support Vector Machine (SVM)
- Long Short-Term Memory network (LSTM)
- Temporal Convolutional Network (TCN)

## 2.Data Provenance and Licensing

- **Original dataset:** `swm_trialA_1K.csv`
- **Original dataset DOI:** [10.26255/healyinb-b1xl](https://doi.org/10.26255/healyinb-b1xl)
- **Original dataset licence:** CC BY-SA 4.0
- **Processed derivative:** `Good_Data.csv`
- **Source-code licence:** MIT

`Good_Data.csv` is a quality-controlled derivative of the original DAIAD smart-meter dataset. The original dataset remains the authoritative source and is not claimed as a dataset created by the authors of this repository.
  
## 3. Repository Structure
```text
CFT-TimeVAE-WaterDemandForecasting/
|
|-- data/
|   |-- raw/
|   |   `-- swm_trialA_1K.csv
|   |
|   `-- Processed_data/
|       |-- Good_Data.csv
|       |-- user_frequency_distribution.py
|       `-- Fig2.png
|
|-- docs/
|   |-- Experiment_Running_Manual.docx
|   `-- Experiment_Running_Manual.pdf
|
|-- outputs/
|   |-- temporal_split_data/
|   |   |-- train/
|   |   |-- val/
|   |   `-- test/
|   |
|   |-- CFT-VAE/
|   |   |-- history/
|   |   |-- metamodels/
|   |   |-- synthetic_data/
|   |   `-- CFT-VAE-forecasting_results/
|   |
|   |-- Time-VAE/
|   |   |-- original_timevae_results/
|   |   `-- Time-vae_forecasting_results/
|   |
|   `-- Visualization/
|
|-- src/
|   |-- Aggregation&Temporal_Preprocessing/
|   |   `-- Aggregation&Temporal_Preprocessing.py
|   |
|   |-- CFT-VAE/
|   |   |-- Model_definition.py
|   |   |-- Trainning.py
|   |   |-- Forcasting.py
|   |   `-- forecasting_lstm_tcn.py
|   |
|   |-- Time-VAE/
|   |   |-- Time_vae_model.py
|   |   |-- Train.py
|   |   |-- Forcasting.py
|   |   `-- forecasting_lstm_tcn_timevae.py
|   |
|   `-- Visulization/
|       |-- baseline forecasting/
|       |-- Direct Fidelity Assessment/
|       |-- Synthetic Data Generation, Forecasting, and Statistical Validation Results/
|       |-- Robustness Analysis of CFT-VAE Augmentation Performance/
|       |-- Distributional Analysis of Augmentation Gains/
|       |-- Computational Performance and Scalability Analysis/
|       `-- Ablation Analysis of the Conditioning Mechanism/
|
|-- .gitattributes
|-- .gitignore
|-- LICENSE
|-- README.md
`-- requirements.txt
```

## 4. Installation

### 1. Install and configure Git LFS

Git LFS is required to download the large CSV and NPZ files.

```powershell
git lfs install
```
### 2. Clone the repository
```powershell
git clone https://github.com/Biniam150/CFT-TimeVAE-WaterDemandForecasting.git
cd CFT-TimeVAE-WaterDemandForecasting
git lfs pull
```
### 3. Install the Python dependencies

```powershell
pip install -r requirements.txt
```
## 5. Running the Full Experimental Pipeline

The complete experimental workflow consists of the following stages:

1. Aggregate the water-demand data and construct the temporal and conditional features.
2. Create the temporal training, validation, and testing datasets.
3. Train CFT-VAE and generate conditional synthetic water-demand data.
4. Evaluate CFT-VAE using four classical forecasting models, LSTM, and TCN.
5. Train the standard multivariate TimeVAE and generate synthetic water-demand data without explicit conditional separation or forecasting-oriented supervision.
6. Evaluate TimeVAE using four classical forecasting models, LSTM, and TCN.
7. Compare real-only, synthetic-only, and augmented forecasting scenarios.
8. Conduct fidelity, statistical-significance, robustness, scalability, and ablation analyses.

Detailed commands, input requirements, expected output files, and troubleshooting instructions are provided in the experiment-running manual:

- [Experiment Running Manual — PDF](docs/Experiment_Running_Manual.pdf)
- [Experiment Running Manual — Word](docs/Experiment_Running_Manual.docx)

Users should follow the stages in the manual sequentially. Aggregation and temporal preprocessing must be completed first because the resulting datasets are shared by the CFT-VAE and TimeVAE pipelines.

## 6: License
MIT License.



