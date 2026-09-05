import pandas as pd
import numpy as np
from mp_api.client import MPRester
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
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
            'nsites',
            'volume',
            'density',
            'density_atomic',
            'symmetry',
            'formation_energy_per_atom',
            'energy_per_atom',
            'energy_above_hull',
            'bulk_modulus',
            'shear_modulus',
            'homogeneous_poisson',
            'has_props',
            'is_metal'
        ],
        is_metal = False,
        num_chunks=160, # 50 chunks of data might be overhelming for some computers, feel free to change it to something >= 10
        chunk_size=1000,
    )
# Convert to DataFrame
df = pd.DataFrame([
    {
        'formula': mat.formula_pretty,
        'volume_per_atom': mat.volume / mat.nsites,
        'density': mat.density,
        'density_atomic': mat.density / mat.nsites,
        'formation_energy_per_atom': mat.formation_energy_per_atom,
        'energy_per_atom': mat.energy_per_atom,
        'energy_above_hull': mat.energy_above_hull,
        'bulk_modulus': mat.bulk_modulus['vrh'] if mat.bulk_modulus else None,
        'shear_modulus': mat.shear_modulus['vrh'] if mat.shear_modulus else None,
        'homogeneous_poisson': mat.homogeneous_poisson if mat.homogeneous_poisson else None,
        'has_elasticity': mat.has_props['elasticity'] if mat.has_props['elasticity'] else False,
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
# Convert crystal_system into 7 columns with 0 or 1 in each column for each crystal system type
#df = pd.get_dummies(df, columns=['crystal_system'])
# Get rid of noise, do not count for some materials having 100+ atoms
df = df[df['volume_per_atom'] < 100]

# ===== ADD THIS =====
# Filter out negative moduli (physically impossible)
df = df[(df['bulk_modulus'] > 0) & (df['shear_modulus'] > 0)]
def remove_outliers_iqr(df, columns, multiplier=1.5):
    """Remove outliers beyond multiplier * IQR from Q3"""
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + multiplier * IQR
        df_clean = df_clean[df_clean[col] <= upper_bound]
    return df_clean

df = remove_outliers_iqr(df, ['bulk_modulus', 'shear_modulus'], multiplier=1.5)

# ===== DATA QUALITY CHECK =====
print("\n=== DATA QUALITY CHECK ===")
print(f"Total materials: {len(df)}")
print(f"\nMissing values:")
print(df[['bulk_modulus', 'shear_modulus', 'homogeneous_poisson']].isnull().sum())
print(f"\nModulus statistics:")
print(df[['bulk_modulus', 'shear_modulus']].describe())

# Check Poisson ratio
print(f"\nHomogeneous Poisson ratio statistics:")
print(df['homogeneous_poisson'].describe())
print(f"Poisson ratio out of physical bounds (-1 to 0.5): {((df['homogeneous_poisson'] < -1) | (df['homogeneous_poisson'] > 0.5)).sum()}")

# Check energy_above_hull distribution
print(f"\nEnergy above hull statistics:")
print(df['energy_above_hull'].describe())
print(f"Percentage of zero values: {(df['energy_above_hull'] == 0).sum() / len(df) * 100:.1f}%")
print(f"Percentage of values < 0.001: {(df['energy_above_hull'] < 0.001).sum() / len(df) * 100:.1f}%")

# Check elasticity distribution
print(f"\nElasticity property distribution:")
print(df['has_elasticity'].value_counts())
print(f"Elasticity balance: {(df['has_elasticity'] == True).sum() / len(df) * 100:.1f}% have elasticity data")


# ===== PREPARE TARGETS =====
# Classification targets
classification_cols = ['has_elasticity']

# Prepare features and target
cols_to_drop = ['formula', 'composition']
regression_cols = ['volume_per_atom',
            'density',
            'density_atomic',
            'formation_energy_per_atom',
            'energy_per_atom',
            'energy_above_hull',
            'bulk_modulus',
            'shear_modulus',
            'homogeneous_poisson',
               ]
# All target columns
target_cols = regression_cols + classification_cols

X = df.drop(cols_to_drop + target_cols, axis=1)
X.columns = X.columns.astype(str)
y_regression = df[regression_cols]
y_classification = df[classification_cols]

print(f"\nFeature matrix shape: {X.shape}")
print(f"Regression targets: {len(regression_cols)}")
print(f"Classification targets: {len(classification_cols)}")
print(f"Total features: {X.shape[1]}")

# ===== TRAIN/TEST SPLIT =====
print("\n=== TRAIN/TEST SPLIT ===")
X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = train_test_split(
    X, y_regression, y_classification,
    test_size=0.2,
    random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")


# ===== SCALE FEATURES =====
print("\n=== FEATURE SCALING ===")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === MULTIOUTPUTREGRESSOR ===
base_reg = RandomForestRegressor(
    n_estimators=400,
    max_features='sqrt',
    min_samples_leaf=2,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

regression_model = MultiOutputRegressor(base_reg)
regression_model.fit(X_train_scaled, y_reg_train)

# Evaluate the model
train_score_reg = regression_model.score(X_train_scaled, y_reg_train)
test_score_reg = regression_model.score(X_test_scaled, y_reg_test)
y_reg_pred = regression_model.predict(X_test_scaled)

print(f"Train R² Score: {train_score_reg:.4f}")
print(f"Test R² Score: {test_score_reg:.4f}")
print("\n--- Prediction Metrics ---")
for i, col in enumerate(regression_cols):
    r2 = r2_score(y_reg_test.iloc[:, i], y_reg_pred[:, i])
    mae = mean_absolute_error(y_reg_test.iloc[:, i], y_reg_pred[:, i])
    print(f"{col:25} | R²: {r2:.4f} | MAE: {mae:.4f}")

# === BULK_MODULUS AND SHEAR_MODULUS MODELS ===
bulk_y_train = y_reg_train['bulk_modulus'].values
bulk_y_test  = y_reg_test['bulk_modulus'].values

shear_y_train = y_reg_train['shear_modulus'].values
shear_y_test  = y_reg_test['shear_modulus'].values

bulk_model = RandomForestRegressor(
    n_estimators=800,
    max_features='sqrt',
    min_samples_leaf=2,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

shear_model = RandomForestRegressor(
    n_estimators=400,
    max_features='sqrt',
    min_samples_leaf=2,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

bulk_model.fit(X_train_scaled, bulk_y_train)
shear_model.fit(X_train_scaled, shear_y_train)

bulk_pred = bulk_model.predict(X_test_scaled)
shear_pred = shear_model.predict(X_test_scaled)

bulk_r2  = r2_score(bulk_y_test, bulk_pred)
bulk_mae = mean_absolute_error(bulk_y_test, bulk_pred)

shear_r2  = r2_score(shear_y_test, shear_pred)
shear_mae = mean_absolute_error(shear_y_test, shear_pred)

print("\n--- Prediction Metrics (separate models: bulk/shear) ---")
print(f"{'bulk_modulus':25} | R²: {bulk_r2:.4f} | MAE: {bulk_mae:.4f}")
print(f"{'shear_modulus':25} | R²: {shear_r2:.4f} | MAE: {shear_mae:.4f}")



# ===== CLASSIFICATION MODEL FOR ELASTICITY =====
print("\n=== TRAINING CLASSIFICATION MODEL (elasticity) ===")
elasticity_classifier = RandomForestClassifier(
    n_estimators=250,
    max_depth=10,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
elasticity_classifier.fit(X_train_scaled, y_class_train['has_elasticity'])

y_elasticity_pred = elasticity_classifier.predict(X_test_scaled)

elasticity_accuracy = accuracy_score(y_class_test['has_elasticity'], y_elasticity_pred)
elasticity_precision = precision_score(y_class_test['has_elasticity'], y_elasticity_pred, zero_division=0)
elasticity_recall = recall_score(y_class_test['has_elasticity'], y_elasticity_pred, zero_division=0)
elasticity_f1 = f1_score(y_class_test['has_elasticity'], y_elasticity_pred, zero_division=0)

print(f"\nClassification Model Performance (has_elasticity):")
print(f"Accuracy:  {elasticity_accuracy:.4f}")
print(f"Precision: {elasticity_precision:.4f}")
print(f"Recall:    {elasticity_recall:.4f}")
print(f"F1-Score:  {elasticity_f1:.4f}")

# Save results to a .joblib file
joblib.dump(regression_model, 'not_metals_regression_model.joblib', compress=3)
joblib.dump(elasticity_classifier, 'not_metals_elasticity_classifier.joblib', compress=3)
joblib.dump(scaler, 'not_metals_scaler.joblib', compress=3)
joblib.dump(bulk_model, 'not_metals_bulk_model.joblib', compress=3)
joblib.dump(shear_model, 'not_metals_shear_model.joblib', compress=3)