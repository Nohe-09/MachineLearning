# Predicción de tarifas de acueducto y alcantarillado - Aguas Regionales EPM

Proyecto integrador de Ciencia de Datos (metodología CRISP-DM). Modelo de **regresión** que predice la tarifa
(en COP) de los servicios de agua potable y agua residual de Aguas Regionales EPM en 11 municipios de Antioquia,
a partir del servicio, el tipo de cargo, la clase de uso, el municipio y el mes.

**Aplicación en línea:** (pegar aquí la URL pública de Streamlit)

## Datos

- Fuente: [Datos Abiertos Colombia - Tarifas de acueducto y aguas residuales, Aguas Regionales EPM](https://www.datos.gov.co/resource/ra8x-nf4u.json) (API ra8x-nf4u).
- 252 filas en formato ancho (una columna por municipio), que pasan a 2.772 observaciones en formato largo y a 2.520
  tras eliminar las tarifas en cero de agua residual en Belén de Bajirá y El Reposo (servicio no registrado).
- Cobertura: enero a julio de 2025.

## Modelo

- Algoritmo final: **XGBoost** dentro de un `Pipeline` de scikit-learn (one-hot de las variables categóricas + XGBRegressor).
- Hiperparámetros: {'n_estimators': 497, 'learning_rate': 0.15695770551890384, 'max_depth': 8, 'subsample': 0.8090931317527976, 'colsample_bytree': 0.885297914889198, 'reg_lambda': 0.9718319944817401}
- Ajuste con `RandomizedSearchCV` (validación cruzada de 5 particiones sobre el 70 % de entrenamiento), métrica RMSE.
- Resultados en el 30 % de prueba: **MAE 28.1 COP, RMSE 52.9 COP, R2 0.9997**.

## Estructura del repositorio

| Archivo | Descripción |
|---|---|
| `app.py` | Aplicación de Streamlit |
| `modelo_tarifa.joblib` | Pipeline completo serializado |
| `requirements.txt` | Dependencias con versiones fijas |
| `datos_nuevos_ejemplo.csv` | Registros de ejemplo para probar el modelo |

## Ejecutar en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Limitaciones

- Solo siete meses de 2025 y una sola empresa. No sirve para proyectar tarifas de meses futuros: ante un mes que no conoce,
  repite el último nivel de tarifa (julio).
- Estrato 4 y Oficial, y Estrato 5 y Comercial, tienen la misma tarifa en los datos; el modelo no los distingue.
- Las métricas son optimistas respecto a un uso real, porque la partición aleatoria deja meses de la misma combinación en
  entrenamiento y prueba. Con un corte temporal, el error es de alrededor de 3 % (el ajuste de abril).
