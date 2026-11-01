"""Geliştirme: hava sütunlarına Hour ve Month eklenince 5 modelin sonucu nasıl değişiyor?
Çalıştır:  python -m src.gelistirme --veri veri
Çıktı: sonuclar/gelistirme_5_model.csv ve modeller/catboost_hava_saat_ay.pkl
"""
import argparse
import joblib
import pandas as pd
from src.ortak import HEDEF, HAVA, EK, veri_yukle, olc, modeller

KUMELER = {'Yalnız hava': HAVA, 'Hava + Saat + Ay': HAVA + EK}


def main(veri):
    train, test = veri_yukle(veri)
    satirlar = []
    for kume_ad, kol in KUMELER.items():
        for ad, m in modeller().items():
            m.fit(train[kol], train[HEDEF])
            r2, rmse, mae = olc(m, test[kol], test[HEDEF])
            satirlar.append([ad, kume_ad, r2, rmse, mae])
            if ad == 'CatBoost' and kume_ad == 'Hava + Saat + Ay':
                joblib.dump(m, 'modeller/catboost_hava_saat_ay.pkl')
    s = pd.DataFrame(satirlar, columns=['Model', 'Özellikler', 'R2_test', 'RMSE_test', 'MAE_test']).round(3)
    print(s)
    s.to_csv('sonuclar/gelistirme_5_model.csv', index=False)


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('--veri', default='veri')
    main(a.parse_args().veri)
