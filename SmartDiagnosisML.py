import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#Load the Breast Cancer data set

data = load_breast_cancer()
X, y = data.data, data.target
labels = data.target_names
feature_names = data.feature_names

#Print the description of the Breast Cancer data set
print(data.DESCR)

print(data.target_names)

#Standardize the data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Add some noise

# Add Gaussian noise to the dataset
noise_factor = 0.5  # You can tweak this value for more/less noise
rng = np.random.default_rng(seed=42)  # Use numpy's random Generator
X_noisy = X_scaled + noise_factor * rng.normal(loc=0.0, scale=1.0, size=X.shape)

# Load the original and noisy datasets into DataFrames
df = pd.DataFrame(X_scaled, columns=feature_names)
df_noisy = pd.DataFrame(X_noisy, columns=feature_names)

# Display the first few rows of the standardized original and noisy data sets for comparison
print("Original Data (First 5 rows):")
print(df.head())

# Display the first few rows of the noisy data set

print("\nNoisy Data (First 5 rows):")
print(df_noisy.head())

#Visualizing the noise content.
plt.figure(figsize=(12, 6))

# Original Feature Distribution (Noise-Free)
plt.subplot(1, 2, 1)
plt.hist(df[feature_names[5]], bins=20, alpha=0.7, color='blue', label='Original')
plt.title('Original Feature Distribution')
plt.xlabel(feature_names[5])
plt.ylabel('Frequency')

# Noisy Feature Distribution
plt.subplot(1, 2, 2)
plt.hist(df_noisy[feature_names[5]], bins=20, alpha=0.7, color='red', label='Noisy') 
plt.title('Noisy Feature Distribution')
plt.xlabel(feature_names[5])  
plt.ylabel('Frequency')

plt.tight_layout()  # Ensures proper spacing between subplots
plt.show()


print("🔧 Optional Enhancements")

plt.figure(figsize=(8, 6))
plt.hist(df[feature_names[5]], bins=20, alpha=0.5, label='Original', color='blue')
plt.hist(df_noisy[feature_names[5]], bins=20, alpha=0.5, label='Noisy', color='red')
plt.title(f'Overlay Histogram: {feature_names[5]}')
plt.xlabel(feature_names[5])
plt.ylabel('Frequency')
plt.legend()
plt.show()

for i in [0, 5, 10]:
    plt.figure(figsize=(8, 4))
    plt.hist(df[feature_names[i]], bins=20, alpha=0.5, label='Original', color='green')
    plt.hist(df_noisy[feature_names[i]], bins=20, alpha=0.5, label='Noisy', color='orange')
    plt.title(f'Overlay Histogram: {feature_names[i]}')
    plt.xlabel(feature_names[i])
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()

#Plots
print("📊 🔧 Optional Enhancements (if you want to explore further):")

plt.figure(figsize=(12, 6))
plt.plot(df[feature_names[5]][:100], label='Original', lw=3)
plt.plot(df_noisy[feature_names[5]][:100], '--', label='Noisy')
plt.title('Zoomed Scaled Feature Comparison (First 100 Samples)')
plt.xlabel('Sample Index')
plt.ylabel(feature_names[5])
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.scatter(range(len(df)), df[feature_names[5]], label='Original', alpha=0.6)
plt.scatter(range(len(df_noisy)), df_noisy[feature_names[5]], label='Noisy', alpha=0.6, marker='x')
plt.title('Point-wise Comparison of Feature Values')
plt.xlabel('Sample Index')
plt.ylabel(feature_names[5])
plt.legend()
plt.tight_layout()
plt.show()

#Scatterplot
plt.figure(figsize=(12, 6))
plt.scatter(df[feature_names[5]], df_noisy[feature_names[5]], alpha=0.6)
# Dynamically calculate the range for the reference line
min_val = min(df[feature_names[5]].min(), df_noisy[feature_names[5]].min())
max_val = max(df[feature_names[5]].max(), df_noisy[feature_names[5]].max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y = x')  # Reference line
plt.title('Scatterplot: Original vs Noisy Feature')
plt.xlabel('Original Feature')
plt.ylabel('Noisy Feature')
plt.legend()
# Calculate the correlation coefficient between the original and noisy feature
correlation = np.corrcoef(df[feature_names[5]], df_noisy[feature_names[5]])[0, 1]
plt.show()

correlation = np.corrcoef(df[feature_names[5]], df_noisy[feature_names[5]])[0, 1]
print(f"Correlation coefficient: {correlation:.4f}")

#Task 1. Split the data, and fit the KNN and SVM models to the noisy training data
# Split the noisy data set into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.3, random_state=42)

# Initialize the models
svm = SVC(kernel='linear', C=1, gamma='scale', random_state=42)  # Explicitly specifying gamma for clarity
# Initialize the KNN model
# The n_neighbors parameter specifies the number of nearest neighbors to consider when making predictions.
knn = KNeighborsClassifier(n_neighbors=5)  # Default n_neighbors is 5
knn = KNeighborsClassifier(n_neighbors=5)  # Default n_neighbors is 5

# Fit the models to the noisy training data
knn.fit(X_train, y_train)
svm.fit(X_train, y_train)

# Evaluate the models on the testing data
y_pred_knn = knn.predict(X_test)
y_pred_svm = svm.predict(X_test)

# Print the accuracy scores
print("\nTesting Accuracy:\n====================")
print(f"KNN: {accuracy_score(y_test, y_pred_knn):.3f}")
print(f"SVM: {accuracy_score(y_test, y_pred_svm):.3f}")

# Print classification reports
print(f"\nClassification Reports:\n{'-'*20}")
print("\nKNN Testing Data:")
print(classification_report(y_test, y_pred_knn))
print("\nSVM Testing Data:")
print(classification_report(y_test, y_pred_svm))

# Generate and plot confusion matrices for testing data
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# KNN Confusion Matrix
sns.heatmap(conf_matrix_knn, annot=True, cmap='Blues', fmt='d', ax=axes[0],
            xticklabels=labels, yticklabels=labels)
axes[0].set_title('KNN Testing Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# SVM Confusion Matrix
sns.heatmap(conf_matrix_svm, annot=True, cmap='Blues', fmt='d', ax=axes[1],
            xticklabels=labels, yticklabels=labels)
axes[1].set_title('SVM Testing Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# Predictions on the training data
y_pred_train_knn = knn.predict(X_train)
y_pred_train_svm = svm.predict(X_train)


# Evaluate the models on the training data
# Accuracy on training data
print(f"KNN Training Accuracy: {accuracy_score(y_train, y_pred_train_knn):.3f}")
print(f"SVM Training Accuracy: {accuracy_score(y_train, y_pred_train_svm):.3f}")


print("\nKNN Training Classification Report:")
print(classification_report(y_train, y_pred_train_knn))

print("\nSVM Training Classification Report:")
print(classification_report(y_train, y_pred_train_svm))
# Task 3. Plot the confusion matrices for the training data

# Compute confusion matrices for training predictions
conf_matrix_knn = confusion_matrix(y_train, y_pred_train_knn)
conf_matrix_svm = confusion_matrix(y_train, y_pred_train_svm)

# Plotting the confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# KNN confusion matrix
sns.heatmap(conf_matrix_knn, annot=True, cmap='Blues', fmt='d', ax=axes[0],
            xticklabels=labels, yticklabels=labels)
axes[0].set_title('KNN Training Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# SVM confusion matrix
sns.heatmap(conf_matrix_svm, annot=True, cmap='Blues', fmt='d', ax=axes[1],
            xticklabels=labels, yticklabels=labels)
axes[1].set_title('SVM Training Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()
