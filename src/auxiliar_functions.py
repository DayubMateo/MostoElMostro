# ================================================================
# 📦 PIPELINE COMPLETO: UNIFICACIÓN, PROCESAMIENTO Y CHECKSUM
# ================================================================

import os
import pandas as pd
import numpy as np
import hashlib
import json
import openmeteo_requests
import requests_cache
import sys
from retry_requests import retry
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.columnas_seleccionadas import COLUMNAS_SELECCIONADAS
from modules.tarifa_electrica import calcular_tarifa_electrica_general, cargos_por_anio, periodos


# ================================================================
# 1️⃣ UNIFICACIÓN DE EXCELS Y GENERACIÓN DEL CSV
# ================================================================

def unir_todos_los_excels_en_un_csv(lista_de_archivos_excel, carpeta_salida, nombre_csv_salida):
    """
    Une hojas de múltiples archivos Excel, filtra por HORA=23:59:00, agrupa por 'DIA'
    y genera un CSV único.
    """
    print("Iniciando proceso de filtrado y unificación...")
    dataframes_filtrados = []

    try:
        os.makedirs(carpeta_salida, exist_ok=True)
    except Exception as e:
        print(f"FATAL: No se pudo crear la carpeta de salida. Motivo: {e}")
        return

    for archivo_excel in lista_de_archivos_excel:
        print(f"\n--- Procesando archivo: {archivo_excel} ---")
        try:
            todas_las_hojas = pd.read_excel(archivo_excel, sheet_name=None)
        except Exception as e:
            print(f"  ERROR: No se pudo leer el archivo. Motivo: {e}")
            continue

        for nombre_hoja, df_hoja in todas_las_hojas.items():
            print(f"  -> Procesando hoja: '{nombre_hoja}'")

            if 'HORA' not in df_hoja.columns or 'DIA' not in df_hoja.columns:
                print(f"     ADVERTENCIA: La hoja '{nombre_hoja}' carece de 'HORA' o 'DIA'. Omitida.")
                continue

            try:
                filtro_hora = df_hoja['HORA'].astype(str).str.strip().str.startswith("23:59").fillna(False)
                df_filtrado = df_hoja[filtro_hora]

                if not df_filtrado.empty:
                    dataframes_filtrados.append(df_filtrado)
            except Exception as e:
                print(f"     ERROR al filtrar hoja '{nombre_hoja}': {e}")

    if not dataframes_filtrados:
        print("\nADVERTENCIA: No se encontró ningún dato con el filtro especificado.")
        return

    df_final = pd.concat(dataframes_filtrados, ignore_index=True)
    df_agregado = df_final.groupby('DIA', as_index=False).sum(numeric_only=True)

    ruta_salida_completa = os.path.join(carpeta_salida, nombre_csv_salida)
    os.makedirs(carpeta_salida, exist_ok=True)
    df_agregado.to_csv(ruta_salida_completa, index=False, encoding='utf-8-sig')

    print(f"\n✅ CSV generado correctamente: {ruta_salida_completa}")


# ================================================================
# 2️⃣ PROCESAMIENTO INICIAL DEL DATASET
# ================================================================

def procesar_dataset(ruta_csv):
    print("\n🔧 Procesando dataset...")

    df = pd.read_csv(ruta_csv)

    # Filtrar columnas relevantes
    df = df[COLUMNAS_SELECCIONADAS]

    # --- Agregar columnas de fecha ---
    df['DIA'] = pd.to_datetime(df['DIA'], errors='coerce')
    df['Anio'] = df['DIA'].dt.year
    df['Mes'] = df['DIA'].dt.month
    df['Dia'] = df['DIA'].dt.day

    dias_semana = {
        'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miercoles',
        'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sabado', 'Sunday': 'Domingo'
    }
    df['Dia_semana'] = df['DIA'].dt.day_name().map(dias_semana)

    # --- Temperatura ambiente ---
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 32.5672, "longitude": -116.6251,
        "start_date": "2020-01-01", "end_date": "2023-12-31",
        "hourly": "temperature_2m"
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()

    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()), inclusive="left"
        ),
        "temperature_2m": hourly_temperature_2m
    }
    hourly_df = pd.DataFrame(data=hourly_data)
    hourly_df["date_local"] = hourly_df["date"].dt.tz_convert("America/Mexico_City")
    hourly_df["DIA"] = hourly_df["date_local"].dt.date

    temperaturas_diarias = (
        hourly_df.groupby("DIA")["temperature_2m"]
        .mean()
        .reset_index()
        .rename(columns={"temperature_2m": "Temperatura_amb"})
    )

    df["DIA"] = pd.to_datetime(df["DIA"]).dt.date
    temperaturas_diarias["DIA"] = pd.to_datetime(temperaturas_diarias["DIA"]).dt.date
    df = df.merge(temperaturas_diarias, on="DIA", how="left")

    # --- Calcular tarifa eléctrica ---
    df = calcular_tarifa_electrica_general(df, periodos, cargos_por_anio)

    # --- Estación del año ---
    def _get_estacion(fecha):
        mes, dia = fecha.month, fecha.day
        if (mes == 12 and dia >= 21) or (mes in [1, 2]) or (mes == 3 and dia < 20):
            return "Invierno"
        elif (mes == 3 and dia >= 20) or mes in [4, 5] or (mes == 6 and dia < 21):
            return "Primavera"
        elif (mes == 6 and dia >= 21) or mes in [7, 8] or (mes == 9 and dia < 23):
            return "Verano"
        else:
            return "Otoño"

    df['estacion'] = pd.to_datetime(df[['Anio', 'Mes', 'Dia']].rename(columns={'Anio': 'year', 'Mes': 'month', 'Dia': 'day'})).apply(_get_estacion)
    df["Frio (Kw)"] = df["Frio (Kw)"].shift(-1)

    # --- Guardar archivo ---
    df.iloc[:-1].to_csv(ruta_csv, index=False, encoding='utf-8-sig')
    print(f"✅ Dataset procesado y guardado en {ruta_csv}")

    return df


# ================================================================
# 3️⃣ VERIFICACIÓN Y GUARDADO DE CHECKSUM
# ================================================================

def verificar_y_guardar_checksum(ruta_dataset, ruta_checksum="../data/checksums.json"):
    os.makedirs(os.path.dirname(ruta_checksum), exist_ok=True)

    def calcular_md5(ruta, buffer_size=65536):
        md5 = hashlib.md5()
        with open(ruta, "rb") as f:
            for bloque in iter(lambda: f.read(buffer_size), b""):
                md5.update(bloque)
        return md5.hexdigest()

    checksum_actual = calcular_md5(ruta_dataset)

    if os.path.exists(ruta_checksum):
        with open(ruta_checksum, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    checksum_guardado = data.get(os.path.basename(ruta_dataset))
    if checksum_guardado == checksum_actual:
        print(f"✅ El dataset '{ruta_dataset}' NO ha cambiado.")
    elif checksum_guardado:
        print(f"⚠️ ATENCIÓN: El dataset '{ruta_dataset}' cambió.")
        print(f"🔸 Anterior: {checksum_guardado}")
        print(f"🔸 Actual:   {checksum_actual}")
    else:
        print(f"ℹ️ No había checksum previo. Se creará uno nuevo.")

    data[os.path.basename(ruta_dataset)] = checksum_actual
    with open(ruta_checksum, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"💾 Checksum actualizado en '{ruta_checksum}'.")


# ================================================================
# 4️⃣ PIPELINE COMPLETO
# ================================================================

def ejecutar_pipeline_completo():
    archivos = [
        "../data/Totalizadores Planta 2022_2023.xlsx",
        "../data/Totalizadores Planta - 2021_2023.xlsx",
        "../data/Totalizadores Planta 2020_2022.xlsx"
    ]
    carpeta_salida = "../data/processed"
    nombre_csv = "dataset_final.csv"
    ruta_csv = os.path.join(carpeta_salida, nombre_csv)

    unir_todos_los_excels_en_un_csv(archivos, carpeta_salida, nombre_csv)
    procesar_dataset(ruta_csv)
    #verificar_y_guardar_checksum(ruta_csv)

    print("\n🚀 Pipeline completo ejecutado con éxito.")


