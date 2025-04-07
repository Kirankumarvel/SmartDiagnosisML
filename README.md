# 🧠 SmartDiagnosisML

SmartDiagnosisML is a machine learning project that compares the performance of **K-Nearest Neighbors (KNN)** and **Support Vector Machines (SVM)** for classifying breast cancer tumors as benign or malignant using the **Breast Cancer Wisconsin** dataset. The project also explores the impact of Gaussian noise on feature data and evaluates model robustness under noisy conditions.

---

## 📊 Dataset

- **Source**: scikit-learn's `load_breast_cancer()` dataset
- **Features**: 30 numeric features (mean, standard error, worst of various cell nuclei measurements)
- **Target**: Binary classification — `malignant (0)` or `benign (1)`

---

## 🔍 Project Highlights

- Feature scaling using **StandardScaler**
- Added **Gaussian noise** to simulate real-world data imperfections
- Visual comparison: original vs noisy features (histograms, line plots, scatterplots)
- Model comparison using:
  - Accuracy Score
  - Classification Report (Precision, Recall, F1-Score)
  - Confusion Matrix
- Trained and tested on both clean and noisy data
- Analysis of **false negatives**, overfitting, and generalization

---

## 🚀 Models Used

| Model | Accuracy (Train) | Accuracy (Test) | False Negatives (Test) |
|-------|------------------|-----------------|--------------------------|
| KNN   | 95.5%            | 93.6%           | 7                        |
| SVM   | 97.2%            | 97.1%           | 2                        |

---

## 📌 Key Findings

- **False Negatives** are the most dangerous in cancer diagnosis — SVM produced significantly fewer.
- **SVM outperforms KNN** across most evaluation metrics.
- SVM showed **better generalization** and less overfitting than KNN on noisy data.

---

## 🛠️ How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/Kirankumarvel/SmartDiagnosisML.git
   cd SmartDiagnosisML
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```


---

## 🧠 Future Improvements

- Add cross-validation for more robust evaluation
- Try additional classifiers (Random Forest, Gradient Boosting, etc.)
- Optimize hyperparameters using GridSearchCV

---

## 📸 Screenshots

<p align="center"> 
  <img src="screenshots/histogram_comparison.png" width="45%"> 
  <img src="screenshots/confusion_matrix.png" width="45%"> 
</p>

---

## 📄 License

MIT License — feel free to use, modify, and contribute!

---

## 💬 Credits

Created by Kiran Kumar  as part of an ML lab exercise.
