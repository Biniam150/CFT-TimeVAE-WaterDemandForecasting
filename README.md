# Conditional Forecasting TimeVAE (CFT-VAE) for Water Demand Forecasting

This repository provides a condtional TimeVAE-based framework for generating synthetic water-demand sequences and evaluating their usefulness for demand forecasting across multiple user-aggregation levels. It includes data preprocessing, conditional feature engineering, CFT-VAE and TimeVAE training, synthetic data generation, forecasting experiments, and statistical validation.

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
- 
## 2. Repository Structure

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
|-- .gitattributes
|-- .gitignore
|-- LICENSE
|-- README.md
`-- requirements.txt
```

## 3. Installation

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

## 4. Step-by-Step: Running the Full Pipeline(Usage)
### Step 1: Run Aggregation + Preprocessing
This script will load Good_Data.csv, perform aggregation for all levels (1 → 1000 users), apply a temporal train/validation/test split, extract conditional features, and save all outputs to outputs/temporal_split_data/.

Run: python src/Aggregation_Temporal_Preprocessing.py

Outputs will appear under:
```plaintext
outputs/temporal_split_data/
├── train/
├── val/
└── test/
```
### Step 2: Train CFT-TimeVAE + Generate Synthetic Data
The script will detect all available trials for the selected level, use NUM_TRIALS to select the first N trials, train TimeVAE for each selected trial, generate synthetic samples based on the chosen ratio, and save model weights, metadata, and synthetic datasets. 

Inside Model_Training.py, adjust: SELECTED_LEVEL = 70, NUM_TRIALS = 5, SELECTED_RATIO = 1.0, NORM_KEY = 'total_demand_clipped', GENERATE_SYNTHETIC = True.

Run: python src/Model_Training.py

Outputs will appear under:
```plaintext
outputs/trained_models/
├── vae_model_level70_trial1.h5
├── vae_model_level70_trial2.h5
├── trial_meta_level70_trial1.npz
├── trial_meta_level70_trial2.npz
└── synthetic_data/
    ├── ratio_1p0/
    │   ├── synthetic_level70_trial1_ratio1p0.npz
    │   └── synthetic_level70_trial2_ratio1p0.npz
```
### Step 3: Forecasting Experiments
The script will run three forecasting tests for all models: baseline (real-only), synthetic-only, and augmented (real + synthetic).

Configure inside Forecasting_Experiment.py: SELECTED_LEVEL = 70, NUM_TRIALS = 5, SELECTED_RATIO = 1.0, NORM_KEY = 'total_demand_clipped'.

Run:python src/Forecasting_Experiment.py

Outputs will appear under:
```plaintext
outputs/forecasting_results/
└── results_level70_ratio1p0.csv
```
### Step 4: Analyze & Visualize Results
This script produces RMSE histograms, summary tables, trial-level statistics, and success-rate comparisons.

Run:python src/Result&Visualization.py

Outputs will appear under:
```plaintext
outputs/visualizations/
├── RMSE_Distribution_Level70_Ratio1p0.png
└── Summary_Tables.txt
```
## 6: License
MIT License.



