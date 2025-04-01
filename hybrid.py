import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from math import pi

# Set style for plots
plt.style.use('seaborn-v0_8')  # Updated style name
sns.set_style("whitegrid")
sns.set_palette("Set2")

# Load dataset
print("Loading data...")
try:
    df = pd.read_csv("loan_dataset.csv")  # Ensure this file exists
    df.dropna(inplace=True)  # Handle missing values
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
rfe = RFE(RandomForestClassifier(n_estimators=1500, max_depth=50, random_state=42), n_features_to_select=35)
X_resampled = rfe.fit_transform(X_resampled, y_resampled)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
print("Training models...")
rf = RandomForestClassifier(n_estimators=1500, max_depth=50, min_samples_split=2, 
                          min_samples_leaf=1, max_features='sqrt', random_state=42)
rf.fit(X_train, y_train)

xgb = XGBClassifier(learning_rate=0.025, n_estimators=1200, max_depth=10, 
                   subsample=0.95, colsample_bytree=0.95, random_state=42, eval_metric='logloss')
xgb.fit(X_train, y_train)

catboost = CatBoostClassifier(iterations=1200, depth=10, learning_rate=0.025, 
                             verbose=0, random_state=42)
catboost.fit(X_train, y_train)

# Voting Classifier
voting_clf = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb), ('catboost', catboost)],
    voting='soft'
)
voting_clf.fit(X_train, y_train)

# Hybrid Model (RF + XGB + CatBoost)
preds_rf = rf.predict_proba(X_test)
preds_xgb = xgb.predict_proba(X_test)
preds_cat = catboost.predict_proba(X_test)
hybrid_preds = (preds_rf * 0.25 + preds_xgb * 0.25 + preds_cat * 0.5)
hybrid_preds = np.argmax(hybrid_preds, axis=1)

# Model evaluations
print("Evaluating models...")
def evaluate_model(y_true, y_pred, y_proba=None):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred)
    }
    if y_proba is not None:
        metrics['ROC-AUC'] = roc_auc_score(y_true, y_proba[:, 1])
    return metrics

model_metrics = {
    "Random Forest": evaluate_model(y_test, rf.predict(X_test), rf.predict_proba(X_test)),
    "XGBoost": evaluate_model(y_test, xgb.predict(X_test), xgb.predict_proba(X_test)),
    "CatBoost": evaluate_model(y_test, catboost.predict(X_test), catboost.predict_proba(X_test)),
    "Voting Classifier": evaluate_model(y_test, voting_clf.predict(X_test), voting_clf.predict_proba(X_test)),
    "Hybrid Model": evaluate_model(y_test, hybrid_preds, (preds_rf * 0.25 + preds_xgb * 0.25 + preds_cat * 0.5))
}

# Convert metrics to DataFrame for easier plotting
metrics_df = pd.DataFrame.from_dict(model_metrics, orient='index')
print("\nModel Performance Metrics:")
print(metrics_df.round(4))

# Visualization 1: Accuracy Comparison Bar Plot
plt.figure(figsize=(12, 6))
models = metrics_df.index
accuracies = metrics_df['Accuracy']

# Sort models by accuracy
sorted_idx = np.argsort(accuracies)[::-1]
models = [models[i] for i in sorted_idx]
accuracies = [accuracies[i] for i in sorted_idx]

bars = plt.bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom')

plt.xlabel("Models", fontsize=12)
plt.ylabel("Accuracy Score", fontsize=12)
plt.title("Model Accuracy Comparison", fontsize=14, pad=20)
plt.xticks(rotation=45, ha='right')
plt.ylim(min(accuracies)-0.05, max(accuracies)+0.05)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('accuracy_comparison.png')
plt.show()

# Visualization 2: Radar Chart for Model Comparison
metrics_for_radar = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
n_metrics = len(metrics_for_radar)

angles = [n / float(n_metrics) * 2 * pi for n in range(n_metrics)]
angles += angles[:1]  # Close the plot

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
plt.xticks(angles[:-1], metrics_for_radar, color='grey', size=10)
ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
plt.ylim(0, 1)

colors = ['b', 'g', 'r', 'c', 'm']
for idx, model in enumerate(metrics_df.index):
    values = metrics_df.loc[model, metrics_for_radar].values.flatten().tolist()
    values += values[:1]  # Close the plot
    ax.plot(angles, values, linewidth=1, linestyle='solid', 
            label=model, color=colors[idx])
    ax.fill(angles, values, alpha=0.1, color=colors[idx])

plt.title('Model Performance Radar Chart', size=15, y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig('radar_chart.png')
plt.show()

# Visualization 3: Metric Comparison Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(metrics_df, annot=True, cmap='YlGnBu', fmt='.3f', linewidths=.5)
plt.title('Model Performance Metrics Comparison', pad=20)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('metrics_heatmap.png')
plt.show()

# Visualization 4: Feature Importance Comparison
print("\nAnalyzing feature importances...")
rf_importances = rf.feature_importances_
xgb_importances = xgb.feature_importances_
cat_importances = catboost.get_feature_importance()

# Normalize importances
rf_importances = rf_importances / rf_importances.sum()
xgb_importances = xgb_importances / xgb_importances.sum()
cat_importances = cat_importances / cat_importances.sum()

# Get top 10 features for each model
top_n = 10
top_features = set()

# Get indices of top features from each model
top_features.update(np.argsort(rf_importances)[-top_n:])
top_features.update(np.argsort(xgb_importances)[-top_n:])
top_features.update(np.argsort(cat_importances)[-top_n:])
top_features = sorted(top_features)

# Prepare data for plotting
feature_names = [f'Feature {i}' for i in top_features]
data = {
    'Feature': feature_names * 3,
    'Importance': np.concatenate([
        rf_importances[top_features],
        xgb_importances[top_features],
        cat_importances[top_features]
    ]),
    'Model': ['Random Forest']*len(top_features) + ['XGBoost']*len(top_features) + ['CatBoost']*len(top_features)
}

df_importances = pd.DataFrame(data)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', hue='Model', data=df_importances, palette='Set2')
plt.xlabel('Normalized Importance Score')
plt.ylabel('Features')
plt.title('Top Feature Importances Across Models', pad=20)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

print("\nAll visualizations saved as PNG files.")