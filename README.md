
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange.svg)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20ULB-lightblue.svg)](https://www.kaggle.com/mlg-ulb/creditcardfraud)

> Research implementation for the paper:  
> **"Credit Card Fraud Detection Using Metaheuristic-Based Feature Selection and Machine Learning Algorithms"**  
> Ahamad Nokib Mozumder, Dr. Nusrat Sharmin, Shariful Billah — *Dept. of CSE, MIST, Dhaka, Bangladesh*

---

## 📌 Overview

This project investigates the use of **metaheuristic algorithms for feature selection** to enhance credit card fraud detection performance. Six nature-inspired optimization algorithms are implemented from scratch and combined with four machine learning classifiers on the highly imbalanced Kaggle ULB credit card dataset.

The core idea: instead of using all 30 features, metaheuristic algorithms intelligently search for the **optimal subset of features** that maximizes fraud detection F1-score while minimizing computational overhead.

---

## 📂 Project Structure

```
credit-card-fraud-detection/
│
├── credit-card-fraud-detection.ipynb   # Main notebook (all experiments)
├── README.md                           # Project documentation
└── requirements.txt                    # Python dependencies
```

> **Note:** The dataset (`creditcard.csv`) is not included in this repo due to size.  
> Download it from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in your Google Drive at `MyDrive/Research/creditcard.csv`.

---

## 🗃️ Dataset

| Property | Details |
|---|---|
| **Name** | Credit Card Fraud Detection (ULB Machine Learning Group) |
| **Source** | [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) |
| **License** | ODC-By (Open Data Commons Attribution) |
| **Transactions** | 284,807 |
| **Features** | 30 (V1–V28 via PCA, Time, Amount) |
| **Fraud cases** | 492 (0.172% — highly imbalanced) |
| **Split** | 80% train / 20% test (stratified) |

---

## ⚙️ Metaheuristic Feature Selection Algorithms

All six algorithms are implemented **from scratch** as Python classes with a unified `fitness_function(binary_mask) → F1` interface:

| Algorithm | Abbr. | Features Selected | Mechanism |
|---|---|---|---|
| Particle Swarm Optimization | PSO | 25 | Sigmoid-binarized velocity update |
| Grey Wolf Optimizer | GWO | 28 | Alpha/beta/delta wolf hierarchy |
| Simulated Annealing | SA | 10 | Probabilistic temperature cooling |
| Ant Colony Optimization | ACO | 4 | Pheromone-guided construction |
| Cuckoo Search Optimization | CSO | 19 | Lévy flight via optunity |
| Genetic Algorithm | GA | 22 | DEAP crossover + mutation |

### Fitness Function
Each algorithm maximizes:
```
fitness(mask) = mean_F1(2-fold CV on X_train[:, mask]) − 0.1 × (|mask| / 30)
```
The cardinality penalty discourages selecting all features, promoting compact subsets.

---

## 🤖 Machine Learning Classifiers

| Classifier | Key Settings |
|---|---|
| Logistic Regression (LR) | `solver='liblinear'`, `class_weight='balanced'` |
| K-Nearest Neighbors (KNN) | `n_neighbors=3`, `weights='distance'` |
| Random Forest (RF) | `n_estimators=100`, `class_weight='balanced'` |
| XGBoost (XGB) | `scale_pos_weight=577`, `eval_metric='logloss'` |

---

## 📊 Results

### Without Feature Selection (Baseline)

| Model | Accuracy | F1 Score | MCC | AUC-ROC |
|---|---|---|---|---|
| Logistic Regression | 0.9755 | 0.1144 | 0.2333 | 0.9721 |
| KNN | 0.9996 | 0.8710 | 0.8720 | 0.9336 |
| Random Forest | 0.9995 | 0.8500 | 0.8550 | 0.9500 |
| XGBoost | 0.9995 | 0.8600 | 0.8600 | 0.9700 |

### Best Results per Metaheuristic Algorithm

#### 🐺 GWO — Best Overall (28 features selected)
| Model | Accuracy | F1 Score | MCC | Brier Score | AUC-ROC |
|---|---|---|---|---|---|
| LR | 0.9755 | 0.1142 | 0.2330 | 0.0237 | 0.9709 |
| KNN | 0.9996 | 0.8757 | 0.8770 | 0.0004 | 0.9336 |
| RF | 0.9995 | 0.8523 | 0.8576 | 0.0004 | 0.9479 |
| **XGB** | **0.9996** | **0.8691** | **0.8692** | **0.0004** | **0.9750** |

#### 🐦 PSO (25 features selected)
| Model | Accuracy | F1 Score | MCC | Brier Score | AUC-ROC |
|---|---|---|---|---|---|
| KNN | 0.9996 | 0.8804 | 0.8821 | 0.0004 | 0.9336 |
| XGB | 0.9995 | 0.8586 | 0.8587 | 0.0004 | 0.9692 |
| RF | 0.9995 | 0.8343 | 0.8401 | 0.0004 | 0.9479 |

#### 🧬 GA — DEAP (22 features selected)
| Model | Accuracy | F1 Score | MCC | Brier Score | AUC-ROC |
|---|---|---|---|---|---|
| KNN | 0.9996 | 0.8634 | 0.8654 | 0.0004 | 0.9437 |
| RF | 0.9995 | 0.8475 | 0.8522 | 0.0004 | 0.9550 |
| XGB | 0.9995 | 0.8550 | 0.8551 | 0.0004 | 0.9680 |

#### 🐜 ACO (4 features selected — most compact)
| Model | Accuracy | F1 Score | MCC | Brier Score | AUC-ROC |
|---|---|---|---|---|---|
| KNN | 0.9992 | 0.7176 | 0.7258 | 0.0008 | 0.8824 |
| RF | 0.9992 | 0.7262 | 0.7361 | 0.0006 | 0.9323 |

> **Key finding:** GWO + XGBoost achieved the best overall performance (F1=0.8691, AUC-ROC=0.9750). PSO + KNN achieved the highest F1 of 0.8804. ACO selected only 4 features yet maintained competitive AUC-ROC (~0.93), demonstrating strong feature compression.

---
