import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition, Element

featurizer = ElementProperty.from_preset("magpie")

app = Flask(__name__)
model = joblib.load('metals_regression_model.joblib')
classifier = joblib.load('metals_elasticity_classifier.joblib')
scaler = joblib.load('metals_scaler.joblib')
metal_indentifier = joblib.load('metal_classifier.joblib')
bulk_modulus = joblib.load('metals_bulk_model.joblib')
shear_modulus = joblib.load('metals_shear_model.joblib')

nm_model = joblib.load('not_metals_regression_model.joblib')
nm_classifier = joblib.load('not_metals_elasticity_classifier.joblib')
nm_scaler = joblib.load('not_metals_scaler.joblib')
nm_bulk_modulus = joblib.load('not_metals_bulk_model.joblib')
nm_shear_modulus = joblib.load('not_metals_shear_model.joblib')

target_regression = ['volume_per_atom',
            'density',
            'density_atomic',
            'formation_energy_per_atom',
            'energy_per_atom',
            #'energy_above_hull',
            #'bulk_modulus',
            #'shear_modulus',
            #'homogeneous_poisson'
            ]

feature_names = featurizer.feature_labels()

expected_features = scaler.get_feature_names_out()
print(f"Expected features: {list(expected_features)}")

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods = ['POST'])
def predict():
    try:
        data = request.get_json()
        formula = data['formula']
        metal_on = data['metal_on']
        is_metal = data['is_metal']
        try:
            composition = Composition(formula)
        except Exception as e:
            return jsonify({'error': f'Invalid formula: {str(e)}', 'status': 'error'}), 400
        features = featurizer.featurize(composition)
        predicting = pd.DataFrame([features], columns=feature_names)
        predicting.columns = predicting.columns.astype(str)
        if not metal_on:
            is_metal = metal_indentifier.predict(predicting)[0]
            predicting["is_metal"] = is_metal
            predicting = predicting[expected_features]
        else:
            predicting["is_metal"] = is_metal
            predicting = predicting[expected_features]

        if is_metal:
            predicting_scaled = scaler.transform(predicting)
            regression_prediction = model.predict(predicting_scaled)[0]
            elasticity_prediction = classifier.predict(predicting_scaled)[0]
            bulk_prediction= bulk_modulus.predict(predicting_scaled)[0]
            shear_prediction = shear_modulus.predict(predicting_scaled)[0]

        else:
            predicting_scaled = nm_scaler.transform(predicting)
            regression_prediction = nm_model.predict(predicting_scaled)[0]
            elasticity_prediction = nm_classifier.predict(predicting_scaled)[0]
            bulk_prediction = nm_bulk_modulus.predict(predicting_scaled)[0]
            shear_prediction = nm_shear_modulus.predict(predicting_scaled)[0]
            print("BB")
        result = {
            'formula': formula,
            'is_metal': bool(is_metal),
            'predictions': {
                name: float(value)
                for (name, value) in zip(target_regression, regression_prediction)
            },
            'has_elasticity': bool(elasticity_prediction),
            'bulk_modulus': float(bulk_prediction),
            'shear_modulus': float(shear_prediction),
            'status': 'success'
        }
        return jsonify(result)
    except Exception as e:
        print(e)
        return jsonify({'error': str(e), 'status': 'error'})

if __name__ == '__main__':
    app.run(debug=True)