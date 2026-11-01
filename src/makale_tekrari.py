"""Makalenin 5 modelini yalnızca hava sütunlarıyla eğitir (varsayılan ayarlar).
Çalıştır:  python -m src.makale_tekrari --veri veri
"""
import argparse
import pandas as pd
from src.ortak import HEDEF, HAVA, veri_yukle, olc, modeller


def main(veri):
    train, test = veri_yukle(veri)
    satirlar = []
    for ad, m in modeller().items():
        m.fit(train[HAVA], train[HEDEF])
        satirlar.append([ad, *olc(m, train[HAVA], train[HEDEF]), *olc(m, test[HAVA], test[HEDEF])])
    s = pd.DataFrame(satirlar, columns=['Model', 'R2_egitim', 'RMSE_egitim', 'MAE_egitim',
                                        'R2_test', 'RMSE_test', 'MAE_test']).round(3)
    print(s)
    s.to_csv('sonuclar/makale_tekrari.csv', index=False)


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('--veri', default='veri')
    main(a.parse_args().veri)
