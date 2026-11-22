# Güneş Enerjisi Tahmini

Nguyen vd. (2025), PLOS ONE makalesinin tekrarı ve "saat + ay" geliştirmesi.
Veri: Williams ve Wagner (2019), Mendeley Data. Hedef: panel gücü (PolyPwr, W).

## Klasörler
- `src/ortak.py`: veri yükleme, ölçütler (R², RMSE, MAE), 5 model
- `src/makale_tekrari.py`: makalenin 5 modeli, yalnızca hava sütunlarıyla
- `src/gelistirme.py`: hava + saat + ay denemesi, CatBoost modelini kaydeder
- `app/demo.py`: kaydırıcılı demo (Streamlit)
- `sonuclar/`: sonuç tabloları
- `veri-....zip`: train.csv ve test.csv (%70 / %30 bölünmüş veri)

## Çalıştırma
1. `veri-....zip` dosyasını çıkarın; proje klasöründe `veri/train.csv` ve `veri/test.csv` olmalı.
2. Kütüphaneleri kurun:
   ```
   pip install -r requirements.txt
   ```
3. Makale tekrarı (isteğe bağlı):
   ```
   python -m src.makale_tekrari --veri veri
   ```
4. Geliştirme ve modeli üretme:
   ```
   python -m src.gelistirme --veri veri
   ```
5. Demo:
   ```
   streamlit run app/demo.py
   ```

## Sonuç (test R²)
| | Yalnız hava | Hava + saat + ay |
|---|---|---|
| CatBoost | 0,551 | 0,662 |
| LightGBM | 0,553 | 0,653 |
| XGBoost | 0,533 | 0,639 |
| GB | 0,535 | 0,625 |
| KNN | 0,470 | 0,572 |
