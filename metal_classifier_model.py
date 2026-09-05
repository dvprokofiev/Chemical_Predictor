import pandas as pd
import numpy as np
from mp_api.client import MPRester
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
import joblib
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up MaterialsProject API connection
with MPRester(os.getenv('API_KEY')) as mpr:
    materials_data = mpr.materials.summary.search(
        fields=[
            "formula_pretty",
            'is_metal'
        ],

        num_chunks=160,
        chunk_size=1000,
    )
# Convert to DataFrame
df = pd.DataFrame([
    {
        'formula': mat.formula_pretty,
        'is_metal': mat.is_metal,
    }
    for mat in materials_data
])

print(f"Materials in DataFrame: {len(df)}")
print(f"Metal materials: {df['is_metal'].sum()}")

# Drop rows with missing values
df = df.dropna()

# Correctly convert formula to data
df = StrToComposition().featurize_dataframe(df, "formula")
# Basic material properties (magpie)
ep_feat = ElementProperty.from_preset(preset_name="magpie")
df = ep_feat.featurize_dataframe(df, col_id="composition")

# ===== PREPARE TARGETS =====
# Classification targets
classification_cols = ['is_metal']

# Prepare features and target
cols_to_drop = ['formula', 'composition']
# All target columns
target_cols = classification_cols

X = df.drop(cols_to_drop + target_cols, axis=1)
X.columns = X.columns.astype(str)
y_classification = df['is_metal']

print("\n=== TARGET VARIABLE DIAGNOSTICS ===")
print(f"y_classification dtype: {y_classification.dtype}")
print(f"Unique values: {y_classification.unique()}")
print(f"Value counts:\n{y_classification.value_counts(dropna=False)}")
print(f"Contains NaN: {y_classification.isna().any()}")

# Remove any remaining NaN values from target
valid_idx = ~y_classification.isna()
X_train = X_train[valid_idx]
X_test = X_test[valid_idx]
y_class_train = y_class_train[valid_idx]
y_class_test = y_class_test[valid_idx]

# Ensure target is boolean/int
y_class_train = y_class_train.astype(int)
y_class_test = y_class_test.astype(int)

print(f"\nAfter cleaning:")
print(f"y_class_train dtype: {y_class_train.dtype}")
print(f"Unique values: {y_class_train.unique()}")

# ===== TRAIN/TEST SPLIT (REVISED) =====
print("\n=== TRAIN/TEST SPLIT ===")
X_train, X_test, y_class_train, y_class_test = train_test_split(
    X, y_classification.astype(int),  # Ensure int type during split
    test_size=0.2,
    random_state=42
)


# ===== CLASSIFICATION MODEL FOR ELASTICITY =====
print("\n=== TRAINING CLASSIFICATION MODEL (elasticity) ===")
metal_classifier = RandomForestClassifier(
    n_estimators=250,
    max_depth=10,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # Handle imbalance
)
metal_classifier.fit(X_train, y_class_train)

y_metal_pred = metal_classifier.predict(X_test)

metal_accuracy = accuracy_score(y_class_test, y_metal_pred)
metal_precision = precision_score(y_class_test, y_metal_pred, zero_division=0)
metal_recall = recall_score(y_class_test, y_metal_pred, zero_division=0)
metal_f1 = f1_score(y_class_test, y_metal_pred, zero_division=0)

print(f"\nClassification Model Performance (is_metal):")
print(f"Accuracy:  {metal_accuracy:.4f}")
print(f"Precision: {metal_precision:.4f}")
print(f"Recall:    {metal_recall:.4f}")
print(f"F1-Score:  {metal_f1:.4f}")


# Save results to a .joblib file
joblib.dump(metal_classifier, 'metal_classifier.joblib', compress=3)

