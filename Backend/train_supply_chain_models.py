import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, mean_absolute_error, make_scorer
)
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings("ignore")

print("==== Supply Chain Delivery Prediction Model Training ====\n")
print(f"Current date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ========== 1. Load dataset ==========
print("\nLoading dataset...")
# Changed path to match new enhanced dataset
df = pd.read_csv("Data/supply_chain_enhanced_dataset (1).csv")
print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

# Display a few column names to verify
print("\nSample columns in dataset:")
print(list(df.columns[:10]))

# ========== 2. Data preprocessing ==========
print("\nPreprocessing data...")

# Convert date columns
date_columns = ['PO_Date', 'DeliveryDueDate', 'ActualDeliveryDate']
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        print(f"Converted {col} to datetime")

# Verify OnScheduleDelivery format
if 'OnScheduleDelivery' in df.columns:
    print(f"OnScheduleDelivery unique values: {df['OnScheduleDelivery'].unique()}")

    # Convert Yes/No to 1/0 if needed
    if df['OnScheduleDelivery'].dtype == object:
        print("Converting OnScheduleDelivery from string to numeric...")
        df['OnScheduleDelivery'] = df['OnScheduleDelivery'].map({'Yes': 1, 'No': 0})
        print(f"Unique values after conversion: {df['OnScheduleDelivery'].unique()}")

# Drop rows with missing values in key columns
initial_rows = df.shape[0]
df = df.dropna(subset=['LeadTimeDays', 'OnScheduleDelivery'])
print(f"Removed {initial_rows - df.shape[0]} rows with missing values in target variables")

# Convert string columns to proper format
categorical_columns = ['ISO9001_Certified', 'MSME_Status', 'VendorCategory', 'TenderType',
                       'MaterialType', 'PriorityFlag', 'InspectionRequired', 'PenaltyClause']
for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# Handle new vendors
if 'IsNewVendor' not in df.columns and 'VendorExperienceYears' in df.columns:
    df['IsNewVendor'] = (df['VendorExperienceYears'] == 0).astype(int)
    print("Created IsNewVendor feature from VendorExperienceYears")
elif 'IsNewVendor' in df.columns and df['IsNewVendor'].dtype == object:
    df['IsNewVendor'] = df['IsNewVendor'].map({'Yes': 1, 'No': 0})
    print("Converted IsNewVendor from Yes/No to 1/0")

# ========== 3. Feature engineering ==========
print("\nPerforming feature engineering...")

# Create seasonal features if not present
if 'OrderMonth' not in df.columns and 'PO_Date' in df.columns:
    df['OrderMonth'] = df['PO_Date'].dt.month
    print("Created OrderMonth feature")

if 'IsMonsoonSeason' not in df.columns and 'OrderMonth' in df.columns:
    df['IsMonsoonSeason'] = df['OrderMonth'].isin([6, 7, 8]).astype(int)
    print("Created IsMonsoonSeason feature")

if 'IsQuarterEnd' not in df.columns and 'OrderMonth' in df.columns:
    df['IsQuarterEnd'] = df['OrderMonth'].isin([3, 6, 9, 12]).astype(int)
    print("Created IsQuarterEnd feature")

# Ensure we have vendor reliability score
if 'VendorReliabilityScore' not in df.columns:
    print("Computing VendorReliabilityScore...")
    # Group by vendor and calculate average on-time performance
    vendor_reliability = df.groupby('VendorID')['OnScheduleDelivery'].mean() * 100
    df['VendorReliabilityScore'] = df['VendorID'].map(vendor_reliability)
    # For vendors with no history, use a lower-than-average value
    median_reliability = df['VendorReliabilityScore'].median()
    df['VendorReliabilityScore'].fillna(median_reliability * 0.8, inplace=True)

# Create material complexity score if not present
if 'MaterialComplexityScore' not in df.columns and 'DelayDays' in df.columns:
    print("Computing MaterialComplexityScore...")
    material_delay_avg = df.groupby('MaterialCode')['DelayDays'].mean()
    df['MaterialComplexityScore'] = df['MaterialCode'].map(material_delay_avg)
    # Normalize to 0-100 scale
    min_score = df['MaterialComplexityScore'].min()
    max_score = df['MaterialComplexityScore'].max()
    df['MaterialComplexityScore'] = ((df['MaterialComplexityScore'] - min_score) /
                                     (max_score - min_score) * 100)
    df['MaterialComplexityScore'].fillna(df['MaterialComplexityScore'].median(), inplace=True)

# ========== 4. Feature selection ==========
print("\nSelecting features...")

# Features for classification (on-time delivery prediction)
class_features = [
    'MaterialCode', 'VendorID', 'PO_Amount', 'ISO9001_Certified',
    'MSME_Status', 'VendorCategory', 'TenderType', 'LeadTimeDays',
    'IsNewVendor'
]

# Additional features if available
additional_class_features = [
    'VendorReliabilityScore', 'MaterialComplexityScore',
    'PastDelayCount', 'PastRejectionRate', 'LogisticsDistanceKM',
    'OrderMonth', 'IsQuarterEnd', 'IsMonsoonSeason', 'PenaltyClause',
    'HoldStatus', 'BlacklistStatus', 'PaymentTerms'
]

# Features for regression (lead time estimation)
reg_features = [
    'MaterialCode', 'VendorID', 'PO_Amount', 'ISO9001_Certified',
    'MSME_Status', 'VendorCategory', 'TenderType',
    'IsNewVendor'
]

# Additional features if available
additional_reg_features = [
    'MaterialComplexityScore', 'LogisticsDistanceKM', 'VendorReliabilityScore',
    'OrderMonth', 'IsQuarterEnd', 'IsMonsoonSeason',
    'VolumeKG', 'MaterialType', 'PriorityFlag', 'InspectionRequired'
]

# Add available additional features
for feat in additional_class_features:
    if feat in df.columns:
        class_features.append(feat)

for feat in additional_reg_features:
    if feat in df.columns:
        reg_features.append(feat)

# Ensure all selected features are in the dataset
class_features = [f for f in class_features if f in df.columns]
reg_features = [f for f in reg_features if f in df.columns]

print(f"Selected {len(class_features)} features for classification:")
print(", ".join(class_features))
print(f"\nSelected {len(reg_features)} features for regression:")
print(", ".join(reg_features))

# Identify categorical and numerical features
categorical_features = [
    'MaterialCode', 'VendorID', 'ISO9001_Certified', 'MSME_Status',
    'VendorCategory', 'TenderType', 'MaterialType', 'PriorityFlag',
    'InspectionRequired', 'PenaltyClause', 'HoldStatus', 'BlacklistStatus',
    'PaymentTerms'
]

categorical_class = [f for f in categorical_features if f in class_features]
numerical_class = [f for f in class_features if f not in categorical_class]
categorical_reg = [f for f in categorical_features if f in reg_features]
numerical_reg = [f for f in reg_features if f not in categorical_reg]

# Target variables
target_class = 'OnScheduleDelivery'
target_reg = 'LeadTimeDays'

# ========== 5. Prepare data for modeling ==========
print("\nPreparing data for modeling...")

# Create directories for outputs
for directory in ['models', 'plots']:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# Time-based split for better evaluation
split_date = pd.to_datetime('2025-01-01')
print(f"Using time-based split with date: {split_date}")

# Preprocess for classification
X_class = df[class_features].copy()
y_class = df[target_class].copy()

if 'PO_Date' in df.columns:
    train_mask = df['PO_Date'] < split_date
    X_train_c, y_train_c = X_class[train_mask], y_class[train_mask]
    X_test_c, y_test_c = X_class[~train_mask], y_class[~train_mask]
    print(f"Classification: {len(X_train_c)} training samples, {len(X_test_c)} testing samples")
else:
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    print(f"Classification: Using random split with {len(X_train_c)} training samples")

# Preprocess for regression
X_reg = df[reg_features].copy()
y_reg = df[target_reg].copy()

if 'PO_Date' in df.columns:
    X_train_r, y_train_r = X_reg[train_mask], y_reg[train_mask]
    X_test_r, y_test_r = X_reg[~train_mask], y_reg[~train_mask]
    print(f"Regression: {len(X_train_r)} training samples, {len(X_test_r)} testing samples")
else:
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    print(f"Regression: Using random split with {len(X_train_r)} training samples")

# ========== 6. Build preprocessing pipelines ==========
print("\nBuilding preprocessing pipelines...")

# Classification preprocessor
preprocessor_class = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_class),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_class)
    ]
)

# Regression preprocessor
preprocessor_reg = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_reg),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_reg)
    ]
)


# ========== 7. Feature Importance Analysis Function ==========
def plot_feature_importance(model, features, filename, top_n=20):
    """Generate feature importance plot for a trained model"""
    if hasattr(model, 'feature_importances_'):
        # Get feature names from the preprocessor
        if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
            try:
                preprocessor = model.named_steps['preprocessor']
                feature_names = preprocessor.get_feature_names_out()
            except:
                # Fallback to simple feature numbers
                feature_names = [f"Feature {i}" for i in range(len(model.feature_importances_))]
        else:
            feature_names = features

        # Sort features by importance
        importances = model.feature_importances_
        if len(importances) > top_n:
            indices = np.argsort(importances)[-top_n:]
        else:
            indices = np.argsort(importances)

        plt.figure(figsize=(12, 8))
        plt.title(f'Top {top_n} Feature Importances')
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)),
                   [feature_names[i] if i < len(feature_names) else f"Feature {i}" for i in indices])
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Feature importance plot saved as {filename}")


# ========== 8. Risk Factor Analysis Function ==========
def identify_risk_factors(row, threshold=0.5):
    """Identify risk factors for a specific prediction"""
    risk_factors = []

    # Lead time analysis
    if row.get('LeadTimeDays', 0) < 4:
        risk_factors.append("CRITICAL: Ultra-short lead time (1-3 days)")
    elif row.get('LeadTimeDays', 0) < 7:
        risk_factors.append("HIGH RISK: Very short lead time (4-6 days)")
    elif row.get('LeadTimeDays', 0) < 15:
        risk_factors.append("Short lead time increases risk")

    # Vendor analysis
    if row.get('IsNewVendor', 0) == 1:
        risk_factors.append("New vendor with no performance history")
    if row.get('ISO9001_Certified', '') == 'No':
        risk_factors.append("Vendor is not ISO9001 certified")
    if row.get('MSME_Status', '') == 'Micro':
        risk_factors.append("Micro MSME vendors have higher delivery variability")

    # Geography
    if row.get('VendorCategory', '') == 'International':
        risk_factors.append("International shipment faces customs and longer transit")

    # Special statuses
    if row.get('BlacklistStatus', '') == 'Yes':
        risk_factors.append("CRITICAL: Vendor is blacklisted")
    if row.get('HoldStatus', '') == 'Yes':
        risk_factors.append("Vendor is on hold status")

    # Seasonal factors
    if row.get('IsMonsoonSeason', 0) == 1:
        risk_factors.append("Monsoon season may cause logistics delays")
    if row.get('IsQuarterEnd', 0) == 1:
        risk_factors.append("Quarter-end rush may affect delivery schedules")

    return risk_factors


# ========== 9. Post-processing adjustment function ==========
def adjust_prediction_probability(probability, lead_time, vendor_category, is_new_vendor, iso_certified, msme_status):
    """Apply business rules to adjust prediction probabilities"""

    # Hard caps for ultra-short lead times
    if lead_time < 4:
        probability = min(probability, 0.30)  # Hard cap at 30% for 1-3 days
    elif lead_time < 7:
        probability = min(probability, 0.50)  # Hard cap at 50% for 4-6 days

    # Hard cap for high risk combinations
    if lead_time < 7 and is_new_vendor == 1 and iso_certified == 'No':
        probability = min(probability, 0.35)

    if lead_time < 7 and vendor_category == 'International' and is_new_vendor == 1:
        probability = min(probability, 0.40)

    if vendor_category == 'International' and is_new_vendor == 1 and iso_certified == 'No' and msme_status == 'Micro':
        probability = min(probability, 0.35)

    return probability


# ========== 10. Train classification model ==========
print("\nTraining classification model for on-time delivery prediction...")

# Pipeline for classification
pipeline_class = Pipeline([
    ('preprocessor', preprocessor_class),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# Expanded parameter grid for hyperparameter tuning
class_param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__max_depth': [3, 5, 7]
}

# Custom scorer that explicitly sets positive label to 1
f1_custom_scorer = make_scorer(f1_score, pos_label=1)

# Perform grid search with custom scorer
grid_class = GridSearchCV(
    pipeline_class, class_param_grid,
    cv=5, scoring=f1_custom_scorer, n_jobs=-1, verbose=1
)

grid_class.fit(X_train_c, y_train_c)
best_class_model = grid_class.best_estimator_

print(f"Best parameters: {grid_class.best_params_}")

# Evaluate classification model
y_pred_c = best_class_model.predict(X_test_c)
y_proba_c = best_class_model.predict_proba(X_test_c)[:, 1]

print("\n===== Classification Metrics =====")
print(f"Accuracy: {accuracy_score(y_test_c, y_pred_c):.4f}")
print(f"Precision: {precision_score(y_test_c, y_pred_c, pos_label=1, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test_c, y_pred_c, pos_label=1, zero_division=0):.4f}")
print(f"F1 Score: {f1_score(y_test_c, y_pred_c, pos_label=1, zero_division=0):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test_c, y_proba_c):.4f}")

# Confusion matrix
cm = confusion_matrix(y_test_c, y_pred_c)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Late', 'On Time'],
            yticklabels=['Late', 'On Time'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('plots/confusion_matrix.png')
plt.close()

# Plot feature importance for classification model
try:
    plot_feature_importance(
        best_class_model.named_steps['classifier'],
        class_features,
        'plots/classification_feature_importance.png'
    )
except Exception as e:
    print(f"Could not generate feature importance plot: {str(e)}")

# ========== 11. Train regression model ==========
print("\nTraining regression model for lead time estimation...")

# Pipeline for regression
pipeline_reg = Pipeline([
    ('preprocessor', preprocessor_reg),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Expanded parameter grid for hyperparameter tuning
reg_param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [10, 20],
    'regressor__min_samples_split': [2, 5]
}

# Perform grid search
grid_reg = GridSearchCV(
    pipeline_reg, reg_param_grid,
    cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1
)

grid_reg.fit(X_train_r, y_train_r)
best_reg_model = grid_reg.best_estimator_

print(f"Best parameters: {grid_reg.best_params_}")

# Evaluate regression model
y_pred_r = best_reg_model.predict(X_test_r)

print("\n===== Regression Metrics =====")
print(f"Mean Absolute Error: {mean_absolute_error(y_test_r, y_pred_r):.4f}")
print(f"RMSE: {np.sqrt(np.mean((y_test_r - y_pred_r) ** 2)):.4f}")
print(f"R² Score: {best_reg_model.score(X_test_r, y_test_r):.4f}")

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test_r, y_pred_r, alpha=0.5)
plt.plot([y_test_r.min(), y_test_r.max()],
         [y_test_r.min(), y_test_r.max()], 'k--', lw=2)
plt.xlabel('Actual Lead Time')
plt.ylabel('Predicted Lead Time')
plt.title('Actual vs Predicted Lead Time')
plt.savefig('plots/lead_time_prediction.png')
plt.close()

# Plot feature importance for regression model
try:
    plot_feature_importance(
        best_reg_model.named_steps['regressor'],
        reg_features,
        'plots/regression_feature_importance.png'
    )
except Exception as e:
    print(f"Could not generate feature importance plot: {str(e)}")

# ========== 12. Analyze risk factors for test set ==========
print("\nAnalyzing risk factors in test data...")

# Create dataframe with test data and predictions
test_results = X_test_c.copy()
test_results['actual_ontime'] = y_test_c
test_results['predicted_ontime'] = y_pred_c
test_results['predicted_probability'] = y_proba_c

# Apply post-processing adjustments to short lead times
test_results['adjusted_probability'] = test_results.apply(
    lambda row: adjust_prediction_probability(
        row['predicted_probability'],
        row['LeadTimeDays'],
        row.get('VendorCategory', ''),
        row.get('IsNewVendor', 0),
        row.get('ISO9001_Certified', 'Yes'),
        row.get('MSME_Status', 'Not MSME')
    ),
    axis=1
)

# Calculate risk factors for each row
test_results['risk_factors'] = test_results.apply(identify_risk_factors, axis=1)

# Count number of risk factors
test_results['num_risk_factors'] = test_results['risk_factors'].apply(len)

# Analyze impact of lead time on prediction accuracy
lead_time_bins = [0, 7, 15, 45, 90, 180]
lead_time_labels = ['Ultra-short (1-7d)', 'Short (8-15d)', 'Normal (16-45d)',
                    'Long (46-90d)', 'Very Long (91+d)']

test_results['lead_time_category'] = pd.cut(
    test_results['LeadTimeDays'],
    bins=lead_time_bins,
    labels=lead_time_labels
)

# Analyze accuracy by lead time category
lead_time_analysis = test_results.groupby('lead_time_category').agg(
    count=('LeadTimeDays', 'count'),
    accuracy=('predicted_ontime', lambda x: accuracy_score(
        test_results.loc[x.index, 'actual_ontime'], x)),
    avg_probability=('predicted_probability', 'mean'),
    avg_adjusted_prob=('adjusted_probability', 'mean'),
    avg_actual=('actual_ontime', 'mean')
)

print("\n===== Accuracy by Lead Time Category =====")
print(lead_time_analysis)

# Plot accuracy by lead time category
plt.figure(figsize=(12, 6))
lead_time_analysis['accuracy'].plot(kind='bar')
plt.title('Prediction Accuracy by Lead Time Category')
plt.ylabel('Accuracy')
plt.xlabel('Lead Time Category')
plt.tight_layout()
plt.savefig('plots/accuracy_by_lead_time.png')
plt.close()

# Compare original vs adjusted probabilities
plt.figure(figsize=(10, 6))
plt.scatter(test_results['predicted_probability'],
            test_results['adjusted_probability'],
            alpha=0.5)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('Original Predicted Probability')
plt.ylabel('Adjusted Probability')
plt.title('Effect of Business Rules on Predicted Probabilities')
plt.savefig('plots/probability_adjustments.png')
plt.close()

# ========== 13. Save models and metadata ==========
print("\nSaving models and metadata...")

# Save classification model
joblib.dump(best_class_model, 'models/ontime_delivery_classifier.pkl')

# Save regression model
joblib.dump(best_reg_model, 'models/lead_time_estimator.pkl')

# Save post-processing functions
with open('models/post_processing.py', 'w') as f:
    f.write("""
# Post-processing functions for supply chain predictions

def adjust_prediction_probability(probability, lead_time, vendor_category, is_new_vendor, iso_certified, msme_status):
    \"\"\"Apply business rules to adjust prediction probabilities\"\"\"

    # Hard caps for ultra-short lead times
    if lead_time < 4:
        probability = min(probability, 0.30)  # Hard cap at 30% for 1-3 days
    elif lead_time < 7:
        probability = min(probability, 0.50)  # Hard cap at 50% for 4-6 days

    # Hard cap for high risk combinations
    if lead_time < 7 and is_new_vendor == 1 and iso_certified == 'No':
        probability = min(probability, 0.35)

    if lead_time < 7 and vendor_category == 'International' and is_new_vendor == 1:
        probability = min(probability, 0.40)

    if vendor_category == 'International' and is_new_vendor == 1 and iso_certified == 'No' and msme_status == 'Micro':
        probability = min(probability, 0.35)

    return probability

def identify_risk_factors(row, threshold=0.5):
    \"\"\"Identify risk factors for a specific prediction\"\"\"
    risk_factors = []

    # Lead time analysis
    if row.get('LeadTimeDays', 0) < 4:
        risk_factors.append("CRITICAL: Ultra-short lead time (1-3 days)")
    elif row.get('LeadTimeDays', 0) < 7:
        risk_factors.append("HIGH RISK: Very short lead time (4-6 days)")
    elif row.get('LeadTimeDays', 0) < 15:
        risk_factors.append("Short lead time increases risk")

    # Vendor analysis
    if row.get('IsNewVendor', 0) == 1:
        risk_factors.append("New vendor with no performance history")
    if row.get('ISO9001_Certified', '') == 'No':
        risk_factors.append("Vendor is not ISO9001 certified")
    if row.get('MSME_Status', '') == 'Micro':
        risk_factors.append("Micro MSME vendors have higher delivery variability")

    # Geography
    if row.get('VendorCategory', '') == 'International':
        risk_factors.append("International shipment faces customs and longer transit")

    # Special statuses
    if row.get('BlacklistStatus', '') == 'Yes':
        risk_factors.append("CRITICAL: Vendor is blacklisted")
    if row.get('HoldStatus', '') == 'Yes':
        risk_factors.append("Vendor is on hold status")

    # Seasonal factors
    if row.get('IsMonsoonSeason', 0) == 1:
        risk_factors.append("Monsoon season may cause logistics delays")
    if row.get('IsQuarterEnd', 0) == 1:
        risk_factors.append("Quarter-end rush may affect delivery schedules")

    return risk_factors
""")

# Save feature lists
model_metadata = {
    "classification_features": class_features,
    "regression_features": reg_features,
    "categorical_features": categorical_features,
    "numerical_features": list(set(numerical_class + numerical_reg)),
    "model_version": "1.0.0",
    "training_date": datetime.now().strftime("%Y-%m-%d"),
    "lead_time_categories": {
        "ultra_short": "1-7 days",
        "short": "8-15 days",
        "normal": "16-45 days",
        "long": "46-90 days",
        "very_long": "91+ days"
    }
}

with open('models/model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)

print("\n✅ Training complete! Models, metadata, and analysis saved successfully.")
print("📊 Check the 'plots' directory for visualizations.")
print("📁 Check the 'models' directory for trained models and metadata.")