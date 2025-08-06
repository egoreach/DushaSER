from catboost import CatBoostClassifier
from joblib import load


stab = load('stab.bin')
model = CatBoostClassifier().load_model('model.cbm', format='cbm')
