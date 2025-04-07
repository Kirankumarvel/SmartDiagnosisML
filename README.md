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

Here is the formatted structure for the screenshots section:


## 📸 Screenshots

- **Original Feature Distribution & Noisy Feature Distribution**
  ![image](https://github.com/user-attachments/assets/aaa42a7b-9b76-48de-bfca-60dc48e90adf)

- **Overlay Histogram: Mean Compactness**
  ![image](https://github.com/user-attachments/assets/4a20570f-a391-4e82-b2b4-c81aa7cdf75d)

- **Overlay Histogram: Mean Radius**
  ![image](https://github.com/user-attachments/assets/ae9918b5-4e95-4f16-a696-4cd46ee605f0)

- **Overlay Histogram: Mean Compactness**
  ![image](https://github.com/user-attachments/assets/050e637a-db4c-4dac-96f0-56781bf2f936)

- **Overlay Histogram: Radius Error**
  ![image](https://github.com/user-attachments/assets/a5e6cb0a-3d59-4d7c-9039-82232ae9a3b1)

- **Zoomed Scaled Feature Comparison (First 100 Samples)**
  ![image](https://github.com/user-attachments/assets/184cb7da-3e9f-413d-ad0c-811454b97963)

- **Point-wise Comparison of Feature Values**
  ![image](https://github.com/user-attachments/assets/348c1dd9-ea06-4c36-9078-356406afa679)

- **Scatterplot: Original vs Noisy Feature**
  ![image](https://github.com/user-attachments/assets/e0bcfbfc-56a5-41da-a240-87e076c64dbc)

- **KNN Testing Confusion Matrix & SVM Testing Confusion Matrix**
  ![image](https://github.com/user-attachments/assets/c461cea9-09a0-4550-ab53-631a77ef5b0d)


## 📄 License

MIT License — feel free to use, modify, and contribute!


## 💬 Credits

Created by Kiran Kumar as part of an ML lab exercise.
