"""Ortak ayarlar ve yardımcı fonksiyonlar. Tüm scriptler buradan okur."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

HEDEF = 'PolyPwr'
HAVA = ['Humidity', 'AmbientTemp', 'Wind.Speed', 'Visibility', 'Pressure', 'Cloud.Ceiling']
EK = ['Hour', 'Month']          # geliştirme: zaman sütunları
SEED = 42


def veri_yukle(klasor='veri'):
    k = Path(klasor)
    return pd.read_csv(k / 'train.csv'), pd.read_csv(k / 'test.csv')


def olc(model, X, y):
    """R2, RMSE, MAE döndürür."""
    p = model.predict(X)
    return r2_score(y, p), float(np.sqrt(mean_squared_error(y, p))), mean_absolute_error(y, p)


def modeller():
    """Makaledeki 5 model, varsayılan ayarlarla."""
    return {
        'GB': GradientBoostingRegressor(random_state=SEED),
        'XGBoost': XGBRegressor(random_state=SEED),
        'KNN': make_pipeline(StandardScaler(), KNeighborsRegressor()),
        'LightGBM': LGBMRegressor(random_state=SEED, verbose=-1),
        'CatBoost': CatBoostRegressor(random_state=SEED, verbose=0),
    }
