import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier  # Added missing import
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from math import pi

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_style("whitegrid")
sns.set_palette("Set2")

# Load dataset
print("Loading data...")
try:
    df = pd.read_csv("loan_dataset.csv")
    df.dropna(inplace=True)
except FileNotFoundError:
    raise FileNotFoundError("The file 'loan_dataset.csv' was not found in the current directory.")

# Encode categorical variables
print("Preprocessing data...")
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']

# Apply SMOTE for oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Feature Engineering - Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_resampled = poly.fit_transform(X_resampled)

# Feature selection using RFE
print("Performing feature selection...")
rfe = RFE(RandomForestClassifier(n_estimators=1500, max_depth=50, random_state=42), n_features_to_select=35)
X_resampled = rfe.fit_transform(X_resampled, y_resampled)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train CatBoost model
print("Training CatBoost model...")
catboost = CatBoostClassifier(iterations=1200, depth=10, learning_rate=0.025, 
                             verbose=0, random_state=42)
catboost.fit(X_train, y_train)

# Model evaluation
print("Evaluating model...")
def evaluate_model(y_true, y_pred, y_proba=None):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_proba[:, 1]) if y_proba is not None else None
    }
    return metrics

metrics = evaluate_model(y_test, catboost.predict(X_test), catboost.predict_proba(X_test))
metrics_df = pd.DataFrame([metrics], index=['CatBoost'])

print("\nCatBoost Performance Metrics:")
print(metrics_df.round(4))

# Visualization 1: Metric Bar Plot
plt.figure(figsize=(10, 6))
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
values = [metrics[m] for m in metrics_to_plot]

bars = plt.bar(metrics_to_plot, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom')

plt.xlabel("Metrics", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.title("CatBoost Model Performance Metrics", fontsize=14, pad=20)
plt.ylim(min(values)-0.05, max(values)+0.05)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('catboost_metrics.png')
plt.show()

# Visualization 2: Confusion Matrix
from sklearn.metrics import confusion_matrix

y_pred = catboost.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted 0', 'Predicted 1'], 
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('CatBoost Confusion Matrix', pad=20)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('catboost_confusion_matrix.png')
plt.show()

# Visualization 3: ROC Curve
from sklearn.metrics import roc_curve, auc

y_proba = catboost.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('CatBoost ROC Curve', pad=20)
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig('catboost_roc_curve.png')
plt.show()

print("\nAll visualizations saved as PNG files.")