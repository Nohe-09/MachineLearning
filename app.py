"""Aplicación de Streamlit: predicción de la tarifa de acueducto y alcantarillado
de Aguas Regionales EPM (11 municipios de Antioquia)."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Tarifas de acueducto - Aguas Regionales EPM", page_icon="💧", layout="centered")

RUTA_MODELO = Path(__file__).parent / "modelo_tarifa.joblib"

MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio"]
CLASES = ["Estrato 1", "Estrato 2", "Estrato 3", "Estrato 4", "Estrato 5", "Estrato 6",
          "Comercial", "Industrial", "Oficial"]
MUNICIPIOS = ["Apartadó", "Belén de Bajirá", "Carepa", "Chigorodó", "El Reposo", "Mutatá", "Olaya",
              "San Jerónimo", "Santa Fe de Antioquia", "Sopetrán", "Turbo"]
# en estos municipios no hay tarifa de agua residual en los datos
SIN_ALCANTARILLADO = ["Belén de Bajirá", "El Reposo"]


@st.cache_resource
def cargar_modelo():
    return joblib.load(RUTA_MODELO)


def formato_cop(valor):
    return "$ " + f"{valor:,.0f}".replace(",", ".")


def armar_entrada(servicio, cargo, clase, municipio, mes):
    return pd.DataFrame([{
        "servicio": servicio, "cargo": cargo, "clase_uso": clase,
        "municipio": municipio, "mes_num": MESES.index(mes) + 1,
    }])


modelo = cargar_modelo()

st.title("💧 Predicción de tarifas de acueducto")
st.write(
    "Modelo de regresión entrenado con las tarifas 2025 de **Aguas Regionales EPM** publicadas en "
    "[Datos Abiertos Colombia](https://www.datos.gov.co/resource/ra8x-nf4u.json). "
    "Elige las características del cobro y el modelo estima la tarifa en pesos colombianos."
)

with st.sidebar:
    st.header("Datos de entrada")
    servicio = st.selectbox("Servicio", ["Agua Potable", "Agua Residual"])
    cargo = st.selectbox("Tipo de cargo", ["Cargo Fijo", "Cargo Variable"])
    clase = st.selectbox("Clase de uso", CLASES, index=2)
    municipio = st.selectbox("Municipio", MUNICIPIOS)
    mes = st.selectbox("Mes (2025)", MESES, index=len(MESES) - 1)

unidad = "por mes" if cargo == "Cargo Fijo" else "por m³ consumido"

if servicio == "Agua Residual" and municipio in SIN_ALCANTARILLADO:
    st.warning(
        f"En {municipio} no hay tarifa de agua residual en los datos (el servicio no aparece registrado). "
        "El modelo no fue entrenado para esta combinación, por eso no se calcula una predicción."
    )
    st.stop()

entrada = armar_entrada(servicio, cargo, clase, municipio, mes)
prediccion = float(modelo.predict(entrada)[0])

st.subheader("Tarifa estimada")
st.metric(label=f"{servicio} · {cargo} · {clase} · {municipio} · {mes}", value=formato_cop(prediccion),
          help="Valor en pesos colombianos (COP)")
st.caption(f"Unidad: COP {unidad}.")

# comparacion entre clases de uso para la misma seleccion
todas = pd.concat([armar_entrada(servicio, cargo, c, municipio, mes) for c in CLASES], ignore_index=True)
comparacion = pd.DataFrame({"Clase de uso": CLASES, "Tarifa estimada (COP)": modelo.predict(todas).round(0)})

st.subheader(f"Comparación por clase de uso en {municipio}")
st.bar_chart(comparacion.set_index("Clase de uso"))
with st.expander("Ver tabla"):
    st.dataframe(comparacion.style.format({"Tarifa estimada (COP)": "{:,.0f}"}), hide_index=True, width="stretch")

with st.expander("Limitaciones del modelo"):
    st.markdown(
        "- Los datos cubren solo **enero a julio de 2025** y una sola empresa (11 municipios de Antioquia).\n"
        "- No se debe usar para proyectar tarifas de meses futuros: el modelo no ha visto un ajuste posterior a abril.\n"
        "- Estrato 4 y Oficial tienen la misma tarifa en los datos, igual que Estrato 5 y Comercial.\n"
        "- Es un modelo de apoyo; la tarifa oficial es la que publica la empresa."
    )
