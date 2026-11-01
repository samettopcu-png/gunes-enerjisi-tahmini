"""Basit demo: hava ve zaman değerlerinden panel gücünü tahmin eder.
Önce:  python -m src.gelistirme --veri veri     (modeli üretir)
Sonra: streamlit run app/demo.py
"""
import joblib
import pandas as pd
import streamlit as st

SUTUNLAR = ['Humidity', 'AmbientTemp', 'Wind.Speed', 'Visibility', 'Pressure', 'Cloud.Ceiling', 'Hour', 'Month']

st.title('Güneş paneli güç tahmini')
st.caption('CatBoost, hava + saat + ay sütunlarıyla eğitildi.')

try:
    model = joblib.load('modeller/catboost_hava_saat_ay.pkl')
except FileNotFoundError:
    st.error('Model dosyası yok. Önce "python -m src.gelistirme --veri veri" komutunu çalıştırın.')
    st.stop()

c1, c2 = st.columns(2)
with c1:
    nem = st.slider('Bağıl nem (%)', 0.0, 100.0, 50.0)
    sic = st.slider('Ortam sıcaklığı (°C)', -10.0, 50.0, 20.0)
    ruz = st.slider('Rüzgar hızı', 0.0, 30.0, 5.0)
    gor = st.slider('Görüş mesafesi', 0.0, 20.0, 10.0)
with c2:
    bas = st.slider('Basınç', 900.0, 1100.0, 1013.0)
    bul = st.slider('Bulut tavanı yüksekliği', 0.0, 100.0, 50.0)
    saat = st.slider('Saat', 0, 23, 12)
    ay = st.slider('Ay', 1, 12, 6)

X = pd.DataFrame([[nem, sic, ruz, gor, bas, bul, saat, ay]], columns=SUTUNLAR)
st.metric('Tahmini panel gücü (W)', f'{float(model.predict(X)[0]):.2f}')
st.caption('Kaydırıcı aralıkları örnek değerlerdir; gerçek veri aralıklarına göre ayarlayın.')
