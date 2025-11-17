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
from datetime import datetime
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.columnas_seleccionadas import COLUMNAS_SELECCIONADAS
from modules.tarifa_electrica import calcular_tarifa_electrica_general, cargos_por_anio, periodos


# ================================================================
# 1️⃣ UNIFICACIÓN DE EXCELS Y GENERACIÓN DEL CSV
# ================================================================

def leer_archivos_excel(lista_de_archivos_excel):
    """
    Lee todos los excels, captura las fechas mínimas de cada archivo.
    Retorna una lista con {'hojas':..., 'primera_fecha':...}
    """
    archivos_leidos = []

    for archivo in lista_de_archivos_excel:
        try:
            hojas = pd.read_excel(archivo, sheet_name=None)
        except Exception as e:
            print(f"ERROR leyendo {archivo}: {e}")
            archivos_leidos.append({'hojas': {}, 'primera_fecha': pd.Timestamp.max})
            continue

        fechas_archivo = []
        for nombre, df in hojas.items():
            if nombre in ['Auxiliar', 'Seguimiento Dia']:
                continue
            if 'DIA' in df.columns:
                fechas = pd.to_datetime(df['DIA'], errors='coerce').dropna()
                if not fechas.empty:
                    fechas_archivo.append(fechas.min())

        primera_fecha = min(fechas_archivo) if fechas_archivo else pd.Timestamp.max
        archivos_leidos.append({'hojas': hojas, 'primera_fecha': primera_fecha})

    return archivos_leidos


def filtrar_hojas_por_hora(archivos_leidos):
    """
    Procesa las hojas, filtrando registros donde HORA empieza con 23:59,
    respetando la fecha límite determinada por el siguiente archivo.
    """
    dataframes_filtrados = []

    for idx, archivo_data in enumerate(archivos_leidos):
        print(f"\n--- Procesando archivo {idx+1}/{len(archivos_leidos)} ---")

        hojas = archivo_data['hojas']
        fecha_limite = (
            archivos_leidos[idx + 1]['primera_fecha']
            if idx < len(archivos_leidos) - 1 else None
        )

        for nombre_hoja, df_hoja in hojas.items():
            if nombre_hoja in ['Auxiliar', 'Seguimiento Dia']:
                continue

            # Validación de columnas
            if 'DIA' not in df_hoja.columns or 'HORA' not in df_hoja.columns:
                continue

            try:
                df_hoja['DIA'] = pd.to_datetime(df_hoja['DIA'], errors='coerce')
                df_hoja['HORA'] = df_hoja['HORA'].astype(str).str.strip()

                # Filtrar últimas horas
                filtro_hora = df_hoja['HORA'].str.startswith("23:59").fillna(False)
                df_filtrado = df_hoja[filtro_hora]

                if df_filtrado.empty:
                    continue

                if fecha_limite is not None:
                    df_filtrado = df_filtrado[df_filtrado['DIA'] < fecha_limite]

                if not df_filtrado.empty:
                    dataframes_filtrados.append(df_filtrado)

            except Exception as e:
                print(f"ERROR procesando hoja '{nombre_hoja}': {e}")

    return dataframes_filtrados


def limpiar_columnas(df_agregado):
    """
    Elimina columnas 'unnamed', columnas con >95% ceros
    y columnas con más de 358 NaN.
    """
    cols_unnamed = df_agregado.columns[df_agregado.columns.str.contains("unnamed", case=False)]
    porcentaje_ceros = (df_agregado == 0).mean() * 100
    cols_ceros_95 = porcentaje_ceros[porcentaje_ceros > 95].index
    cols_nan_358 = df_agregado.columns[df_agregado.isna().sum() > 358]

    cols_a_eliminar = list(set(cols_unnamed).union(cols_ceros_95).union(cols_nan_358))
    df_agregado = df_agregado.drop(columns=cols_a_eliminar)

    print("Columnas eliminadas:", cols_a_eliminar)
    return df_agregado


def unir_todos_los_excels_en_un_csv(lista_de_archivos_excel, carpeta_salida, nombre_csv_salida):
    print("Iniciando proceso de filtrado y unificación...")

    archivos_leidos = leer_archivos_excel(lista_de_archivos_excel)
    dataframes_filtrados = filtrar_hojas_por_hora(archivos_leidos)

    if not dataframes_filtrados:
        print("No se encontró ningún dato filtrado.")
        return

    df_final = pd.concat(dataframes_filtrados, ignore_index=True)
    print("Pre agregado:", df_final.shape)

    df_agregado = df_final.groupby('DIA', as_index=False).mean(numeric_only=True)
    print("Post agregado:", df_agregado.shape)

#    df_agregado = limpiar_columnas(df_agregado)
    df_agregado = df_agregado[COLUMNAS_SELECCIONADAS]

    os.makedirs(carpeta_salida, exist_ok=True)
    ruta_salida = os.path.join(carpeta_salida, nombre_csv_salida)
    df_agregado.to_csv(ruta_salida, index=False, encoding='utf-8-sig')

    print("CSV generado correctamente:", ruta_salida)


# ================================================================
# 2️⃣ FUNCIONES DE PREPROCESSING
# ================================================================

def agregar_columnas_temporales(df):
    df['DIA'] = pd.to_datetime(df['DIA'], errors='coerce')
    df['Anio'] = df['DIA'].dt.year
    df['Mes'] = df['DIA'].dt.month
    df['Dia'] = df['DIA'].dt.day
    
    return df


def agregar_dia_semana(df):
    dias_semana = {
        'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miercoles',
        'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sabado', 'Sunday': 'Domingo'
    }
    df['Dia_semana'] = df['DIA'].dt.day_name().map(dias_semana)
    df["es_finde"] = df["Dia_semana"].isin(["Sabado", "Domingo"]).astype(int)
    return df


def obtener_temperaturas_por_dia(dmin, dmax):
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": 32.5672,
        "longitude": -116.6251,
        "start_date": dmin.strftime("%Y-%m-%d"),
        "end_date": dmax.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m"
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()

    hourly_df = pd.DataFrame({
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        ),
        "temperature_2m": hourly_temperature_2m,
    })

    hourly_df["date_local"] = hourly_df["date"].dt.tz_convert("America/Mexico_City")
    hourly_df["DIA"] = hourly_df["date_local"].dt.date

    return (
        hourly_df.groupby("DIA")["temperature_2m"]
        .mean()
        .reset_index()
        .rename(columns={"temperature_2m": "Temperatura_amb"})
    )


def agregar_temperatura(df):
    # Convertir siempre a datetime ANTES DE NADA
    df["DIA"] = pd.to_datetime(df["DIA"], errors="coerce")

    dmin = df["DIA"].min()
    dmax = df["DIA"].max()

    temperaturas_diarias = obtener_temperaturas_por_dia(dmin, dmax)

    # Asegurar que ambos lados están en datetime
    temperaturas_diarias["DIA"] = pd.to_datetime(temperaturas_diarias["DIA"], errors="coerce")

    # MERGE correcto
    df = df.merge(temperaturas_diarias, on="DIA", how="left")

    return df



def agregar_tarifa(df):
    return calcular_tarifa_electrica_general(df, periodos, cargos_por_anio)


def agregar_estacion(df):
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

    df["estacion"] = df["DIA"].apply(_get_estacion)
    return df


def aplicar_lags(df: pd.DataFrame, columnas: list, n_lags: int = 1):
   
    if not columnas:
        raise ValueError("Debes especificar al menos una columna para aplicar lags.")

    df_out = df.copy()

    for col in columnas:
        if col not in df_out.columns:
            raise ValueError(f"La columna '{col}' no existe en el dataset.")

        for lag in range(1, n_lags + 1):
            df_out[f"{col}_lag{lag}"] = df_out[col].shift(lag)

    return df_out


import re
import pandas as pd

def agregar_ratio_producido_meta(df):
    """
    Crea columnas ratio entre producido y meta SOLO si la columna producida
    contiene '/ Hl' en su nombre (evita totalizadores).

    - Detecta columnas que comienzan con 'Meta'.
    - Extrae la variable base (quitando unidades).
    - Busca una columna producida que contenga la base y además '/ Hl'.
    - Si cumple, crea ratio = producido/meta.

    Retorna:
        df modificado con nuevas columnas de ratio.
    """

    columnas = df.columns.tolist()

    def extraer_variable_base(col_meta):
        base = col_meta.replace("Meta", "", 1).strip()
        base = re.sub(r"\(.*?\)", "", base).strip()  # quita paréntesis y unidades
        return base

    for col in columnas:
        if col.startswith("Meta"):

            col_meta = col
            base = extraer_variable_base(col_meta).lower()

            # Buscar solo columnas producidas que:
            # 1. Contengan el nombre base
            # 2. TENGAN "/ Hl" → indispensables
            posibles = [
                c for c in columnas
                if base in c.lower()
                and not c.startswith("Meta")
                and "/ hl" in c.lower()     # ⭐ filtro clave
            ]

            if not posibles:
                print(f"⚠️ No se encontró producción válida (con '/ Hl') para '{col_meta}'. Saltando.")
                continue

            if len(posibles) > 1:
                print(f"⚠️ Varias columnas con '/ Hl' para '{col_meta}': {posibles}. Usando la primera.")

            col_prod = posibles[0]

            # Crear nombre ratio
            nombre_ratio = f"ratio_{col_prod}"

            # Evitar división por cero
            df[nombre_ratio] = df[col_prod] / df[col_meta].replace({0: pd.NA})

            print(f"✅ Ratio generado: {nombre_ratio} = {col_prod} / {col_meta}")

    return df




def agregar_target(df):
    df["Frio (Kw) tomorrow"] = df["Frio (Kw)"].shift(-1)
    return df.iloc[:-1]


# ================================================================
# 🔧 FUNCIÓN GENERAL DE PREPROCESSING
# ================================================================

def preprocess_general(df):
    df = agregar_columnas_temporales(df)
    df = agregar_dia_semana(df)
    df = agregar_temperatura(df)
    df = agregar_tarifa(df)
    df = agregar_estacion(df)
    df = agregar_ratio_producido_meta(df)
    columnas_lag = [col for col in df.columns if col.strip().lower().endswith("(kw)".lower())] #todas las que terminan en kw
    df = aplicar_lags(df, columnas_lag, n_lags=3)
    # promedio temporal
    df = agregar_target(df)
    return df


# ================================================================
# 3️⃣ PROCESAMIENTO INICIAL DEL DATASET
# ================================================================

def procesar_dataset(ruta_csv):
    print("\n🔧 Procesando dataset...")

    df = pd.read_csv(ruta_csv)
    df = preprocess_general(df)

    df.to_csv(ruta_csv, index=False, encoding='utf-8-sig')
    print(f"✅ Dataset procesado y guardado en {ruta_csv}")
    return df


# ================================================================
# 4️⃣ PIPELINE COMPLETO
# ================================================================

def importar_datos_completo(
    archivos = [
        "data/Totalizadores Planta 2020_2022.xlsx",
        "data/Totalizadores Planta - 2021_2023.xlsx",        
        "data/Totalizadores Planta 2022_2023.xlsx"
    ],
    nombre_csv = "dataset_final.csv"
    ):

    carpeta_salida = "data/processed"
    ruta_csv = os.path.join(carpeta_salida, nombre_csv)

    unir_todos_los_excels_en_un_csv(archivos, carpeta_salida, nombre_csv)
    procesar_dataset(ruta_csv)

    print("\n🚀 Pipeline completo ejecutado con éxito.")


def verificar_y_guardar_checksum(ruta_dataset, ruta_checksum="data/checksums.json"):
    os.makedirs(os.path.dirname(ruta_checksum), exist_ok=True)

    def calcular_md5(ruta, buffer_size=65536):
        md5 = hashlib.md5()
        with open(ruta, "rb") as f:
            for bloque in iter(lambda: f.read(buffer_size), b""):
                md5.update(bloque)
        return md5.hexdigest()

    checksum_actual = calcular_md5(ruta_dataset)
    timestamp_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    nombre_dataset = os.path.basename(ruta_dataset)

    # Cargar historial previo (con manejo de errores)
    if os.path.exists(ruta_checksum):
        try:
            with open(ruta_checksum, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                # si el JSON no es dict, normalizarlo
                print("WARN: formato inesperado en JSON de checksums — se va a normalizar.")
                data = {}
        except Exception as e:
            print(f"ERROR leyendo {ruta_checksum}: {e}. Se inicializa nuevo historial.")
            data = {}
    else:
        data = {}

    # Asegurar que exista la entrada para este dataset y que sea una lista
    if nombre_dataset not in data:
        data[nombre_dataset] = []

    historial = data[nombre_dataset]

    # Normalizar elementos antiguos (por ejemplo si eran strings)
    historial_normalizado = []
    for i, item in enumerate(historial):
        if isinstance(item, dict):
            # ya es lo esperado, conservar
            historial_normalizado.append(item)
        elif isinstance(item, str):
            # entrada antigua que sólo guardaba checksum como string -> convertir
            historial_normalizado.append({"timestamp": None, "checksum": item})
        else:
            # otro tipo inesperado: convertimos a string por seguridad
            historial_normalizado.append({"timestamp": None, "checksum": str(item)})

    # Re-asignar historial ya normalizado
    data[nombre_dataset] = historial_normalizado
    historial = data[nombre_dataset]

    # Comparar con el último (si existe)
    if historial:
        ultimo_entry = historial[-1]
        # seguridad: si la última entrada no tiene 'checksum', tomar su str
        ultimo_checksum = ultimo_entry.get("checksum") if isinstance(ultimo_entry, dict) else str(ultimo_entry)

        if ultimo_checksum == checksum_actual:
            print(f"✅ El dataset '{ruta_dataset}' NO ha cambiado desde el último registro.")
        else:
            print(f"⚠️ ATENCIÓN: El dataset '{ruta_dataset}' cambió.")
            print(f"🔸 Anterior: {ultimo_checksum}")
            print(f"🔸 Actual:   {checksum_actual}")
    else:
        print(f"ℹ️ No había registros previos para '{nombre_dataset}', iniciando historial.")

    # Agregar nueva entrada al historial
    data[nombre_dataset].append({
        "timestamp": timestamp_actual,
        "checksum": checksum_actual
    })

    # Guardar JSON (con backup opcional)
    try:
        # opcional: hacer backup del archivo previo
        if os.path.exists(ruta_checksum):
            try:
                os.replace(ruta_checksum, ruta_checksum + ".bak")
            except Exception:
                # si falla el backup, seguimos igual
                pass

        with open(ruta_checksum, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"💾 Historial actualizado en '{ruta_checksum}'.")
    except Exception as e:
        print(f"ERROR guardando el historial de checksums: {e}")