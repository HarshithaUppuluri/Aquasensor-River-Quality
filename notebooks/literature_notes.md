# Literature Review Notes
## AquaSensor River Water Quality Project

---

## Paper 1 — Jarwar et al. (2025) ← PRIMARY PAPER

Full reference:
Jarwar, M. A., Ul Hasan, N., Boisvert, C., Wheway, P.,
& Faulks, M. R. (2025). IoT-based digital twin for
freshwater pollution monitoring: A use case of River
Derwent. In IEEE PIMRC 2025. IEEE.
https://shura.shu.ac.uk/36324/

What they studied:
[Write in your own words — 2 sentences]

What they found:
[Write in your own words — 2 sentences]

What variables they used:
- Temperature
- Dissolved Oxygen

What limitations they admitted:
[List from the paper]

What future work they suggested:
[List from the paper]

Gaps my project fills from this paper:
1. They used fuzzy logic thresholds — I use ML prediction
2. No weather data — I include sunshine, air temp, rainfall
3. No 15-min prediction — I predict DO 15 min ahead
4. No anomaly detection using residuals — I do this
5. No SHAP explainability — I use SHAP
6. No interactive dashboard — I build this

Director's note (from email reply 06/06/2026):
"Currently AI is used off the shelf — working through 
building up intelligence in AI learning"
This confirms my project extends their work with 
proper ML implementation.

---

## Paper 2 — Heddam & Kisi (2018)

Full reference:
Heddam, S., & Kisi, O. (2018). Modelling daily dissolved
oxygen concentration using least square support vector
machine, multivariate adaptive regression splines and
M5 model tree. Journal of Hydrology, 559, 499–509.
https://doi.org/10.1016/j.jhydrol.2018.02.017

What they studied:
[Write in your own words]

What they found:
[Write in your own words]

Key finding for my project:
Temperature is the strongest predictor of DO.
This confirms what my lecturer said about correlation
between temperature, air temperature and sunshine vs DO.

Gaps my project fills:
1. Daily averages only — I use every 12 minutes
2. Single station — I have 3 sensors
3. No weather variables — I include sunshine and air temp
4. No real-time deployment — I deploy on Streamlit
5. No anomaly detection — I have this

---

## Paper 3 — Ahmed et al. (2019)

Full reference:
Ahmed, U., Mumtaz, R., Anwar, H., Shah, A. A.,
Irfan, R., & García-Nieto, J. (2019). Efficient water
quality prediction using supervised machine learning.
Water, 11(11), 2210.
https://doi.org/10.3390/w11112210

What they studied:
[Write in your own words]

What they found:
Random Forest achieves ~92% accuracy for DO prediction.
This justifies my choice of Random Forest as baseline.

Gaps my project fills:
1. Static dataset — I use live API data
2. No real-time deployment — I deploy on Streamlit
3. No explainability — I use SHAP
4. No weather variables — I include these
5. No anomaly detection — I have this
6. No 15-min ahead prediction — I do this


---

## Paper 4 — Wang et al. (2022)

Full reference:
Wang, Y., Zhou, J., Chen, K., Wang, Y., & Liu, L.
(2022). Water quality prediction method based on LSTM
neural network. In ISKE 2017 (pp. 1–6). IEEE.
https://doi.org/10.1109/ISKE.2017.8258814

What they studied:
[Write in your own words]

What they found:
LSTM outperforms traditional models for time-series DO.

Gaps my project fills:
1. Single station — I have 3 sensors
2. No anomaly detection — I have this
3. No weather variables — I include these
4. No SHAP explainability — I use SHAP
5. No real-time deployment — I deploy on Streamlit


---

## Paper 5 — Molnar (2022)

Full reference:
Molnar, C. (2022). Interpretable machine learning:
A guide for making black box models explainable
(2nd ed.).
https://christophm.github.io/interpretable-ml-book/

What SHAP does:
[Write in your own words — 2 sentences]

Why SHAP matters for my project:
The AquaSensor director needs to explain alerts to
water company customers. SHAP shows exactly which
variable caused the alert — rainfall, temperature,
sunshine etc.

Director confirmed this need (email 06/06/2026):
"predicting pollutants based on river outputs of
simple measures, alongside other complex phenomena
such as weather"
SHAP makes these predictions explainable.


---

## Paper 6 — Khullar & Singh (2022)

Full reference:
Khullar, S., & Singh, N. (2022). Water quality
assessment of a river through artificial intelligence
and machine learning. Journal of Soft Computing in
Civil Engineering, 6(1), 69–90.
https://doi.org/10.22115/SCCE.2022.283708.1337

What they found:
No current system deploys ML for real-time anomaly
detection on live IoT river feeds.
This is the PRIMARY GAP my entire project fills.

Gaps my project fills:
1. No deployed ML on live IoT feeds — I do this
2. No user-friendly dashboard — I build this
3. No SHAP explainability — I use SHAP


---

## Gap Analysis

Based on reading all 6 papers, my project fills
these confirmed gaps:

GAP 1 — No ML prediction on real AquaSensor field data
From: Jarwar et al. (2025)
My solution: ML models trained on live AquaSensor API

GAP 2 — No weather variables in DO prediction
From: Heddam & Kisi (2018), Ahmed et al. (2019)
My solution: Sunshine, air temp, rainfall, wind speed
(confirmed by lecturer 05/06/2026)

GAP 3 — No 15-minute ahead DO prediction
From: Jarwar et al. (2025) only uses current readings
My solution: Predict DO 15 minutes ahead
(confirmed by AquaSensor director 05/06/2026)

GAP 4 — No anomaly detection on live IoT data
From: Khullar & Singh (2022)
My solution: Compare actual vs predicted DO → alert

GAP 5 — No SHAP explainability for river DO alerts
From: All papers — black box models only
My solution: SHAP module explains every alert

GAP 6 — No interactive deployed dashboard
From: All papers — no deployment
My solution: Streamlit dashboard evaluated by users