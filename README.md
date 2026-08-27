# AquaPulse – River Intelligence

AquaPulse is a machine-learning-based river water quality monitoring system developed to support the monitoring of the River Derwent.

The system combines AquaSensor measurements, environmental data, dissolved oxygen forecasting, anomaly detection and explainable artificial intelligence within an interactive web application.

## About AquaPulse

Traditional river monitoring primarily provides information about current or historical river conditions. AquaPulse extends this approach by using machine learning to provide information about possible future changes in dissolved oxygen levels.

The system uses high-frequency AquaSensor measurements together with environmental and weather information to monitor river conditions and generate dissolved oxygen forecasts.

AquaPulse provides an accessible web interface where users can explore current river conditions, historical measurements, future dissolved oxygen forecasts and abnormal river behaviour.

## Key Features

- Live River Derwent water-quality monitoring
- AquaSensor live data integration
- Dissolved Oxygen (DO) monitoring in mg/L
- Dissolved Oxygen percentage monitoring
- Water temperature monitoring
- DO forecasting from 15 to 120 minutes ahead
- Linear Regression, Random Forest and XGBoost model evaluation
- Anomaly detection for unusual river conditions
- SHAP explainability for machine-learning predictions
- Interactive actual and predicted DO trend visualisations
- Historical river data records
- Environmental and weather data integration

## Dissolved Oxygen Forecasting

AquaPulse predicts dissolved oxygen levels at eight forecasting horizons:

- 15 minutes
- 30 minutes
- 45 minutes
- 60 minutes
- 75 minutes
- 90 minutes
- 105 minutes
- 120 minutes

Linear Regression, Random Forest and XGBoost models were developed and evaluated. Model performance is assessed using metrics including MAE, RMSE and R².

The live AquaPulse application uses the selected best-performing models to generate forecasts from the latest available river measurements.

## River Monitoring

The application monitors AquaSensor stations associated with the River Derwent and combines current sensor readings with environmental information.

The web application provides four main sections:

- **Home** – current river conditions and live monitoring information
- **Insights** – interactive visualisations of actual dissolved oxygen, predicted dissolved oxygen and water temperature
- **Data Records** – river monitoring and prediction records
- **About** – information about AquaPulse and the project

## Anomaly Detection

AquaPulse includes anomaly detection to identify potentially unusual river conditions.

This allows the system to complement forecasting with alerts that can help users identify changes in river behaviour requiring further investigation.

## Explainable AI

SHAP explainability is incorporated into the project to support interpretation of machine-learning predictions.

This provides insight into how different environmental and sensor variables influence dissolved oxygen forecasting.

## Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Plotly
- AquaSensor API
- Open-Meteo API
- HTML
- CSS

## Project Purpose

The purpose of AquaPulse is to investigate how machine learning, anomaly detection and explainable AI can support a more proactive and interpretable approach to river water-quality monitoring.

The project demonstrates how live environmental measurements and predictive machine learning can be combined within a web-based monitoring system to examine current conditions, future dissolved oxygen levels and abnormal river behaviour.

## Author

Developed as part of a Computing Research Project at Sheffield Hallam University.