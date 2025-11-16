# Estructura requerida del script
import pandas as pd
import joblib
import sys
import json
import os
from preprocessing_pipeline import construir_pipeline
from src.auxiliar_functions import importar_datos_completo


def load_model(version="latest"):
    """Carga un modelo desde models/model_registry.json.
       - version='latest' → carga la versión más alta (semánticamente)
       - version='vX.Y.Z' → carga esa versión exacta
    """
    
    registry_path = "models/model_registry.json"

    if not os.path.exists(registry_path):
        raise FileNotFoundError("❌ No existe models/model_registry.json")

    # Leer archivo
    with open(registry_path, "r") as f:
        registry = json.load(f)

    if not isinstance(registry, list) or len(registry) == 0:
        raise ValueError("❌ El registry está vacío o mal formado.")

    # -----------------------------
    # Función auxiliar: parse semver
    # -----------------------------
    def parse_version(entry):
        try:
            return tuple(map(int, entry["version"].lstrip("v").split(".")))
        except:
            raise ValueError(f"Versión inválida en entry: {entry}")

    # -----------------------------
    # Caso 1: version == "latest"
    # -----------------------------
    if version == "latest":
        latest_entry = max(registry, key=parse_version)

        if "model_path" not in latest_entry:
            raise KeyError("❌ 'model_path' no encontrado en el registry.")

        model_path = latest_entry["model_path"]
        print(f"📦 Cargando modelo versión {latest_entry['version']} → {model_path}")
        return joblib.load(model_path)

    # -----------------------------
    # Caso 2: versión específica
    # -----------------------------
    for entry in registry:
        if entry["version"] == version:
            if "model_path" not in entry:
                raise KeyError(f"❌ 'model_path' no encontrado en entry de versión {version}")
            model_path = entry["model_path"]
            print(f"📦 Cargando modelo versión {version} → {model_path}")
            return joblib.load(model_path)

    raise ValueError(f"❌ No se encontró la versión '{version}' en el registry")


def load_and_preprocess(filepath):
    """ Carga archivo Excel y aplica todo el preprocesamiento del script
    preprocessing_pipeline.py """
    
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
    importar_datos_completo()
    
    # -----------------------------
    # 1️⃣ Cargar archivo (CSV o Excel)
    # -----------------------------
    extension = os.path.splitext(filepath)[1].lower()

    if extension in [".xlsx", ".xls"]:
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath, sep=",", decimal=".")

    # Asegurar orden por día si corresponde
    if "DIA" in df.columns:
        try:
            df["DIA"] = pd.to_datetime(df["DIA"])
            df = df.sort_values(by="DIA", ignore_index=True)
        except:
            print("⚠ No se pudo convertir 'DIA' a datetime")

    print(f"▶ Shape original cargado: {df.shape}")

    # Guardar fechas (por si el modelo necesita saber a qué día corresponde cada fila)
    dates = df["DIA"] if "DIA" in df.columns else None
    hours = df["HORA"] if "HORA" in df.columns else None

    # -----------------------------
    # 2️⃣ Crear DIA_DEL_ANIO
    # -----------------------------
    if "DIA" in df.columns:
        df["DIA_DEL_ANIO"] = df["DIA"].dt.dayofyear

    # -----------------------------
    # 3️⃣ Rolling moving average
    # -----------------------------
    if "Frio (Kw)" in df.columns:
        df["Frio (Kw)_movil_5"] = df["Frio (Kw)"].rolling(window=5, min_periods=1).mean()

    # -----------------------------
    # 4️⃣ Finde = 1 si Sabado/Domingo
    # -----------------------------
    if "Dia_semana" in df.columns:
        df["finde"] = df["Dia_semana"].isin(["Sabado", "Domingo"]).astype(int)

    # -----------------------------
    # 5️⃣ Separar target
    # -----------------------------
    target = "Frio (Kw) tomorrow"

    if target not in df.columns:
        raise ValueError(f"El dataset no contiene la columna target '{target}'")

    y = df[target]

    # -----------------------------
    # 6️⃣ Construir X sin el target ni DIA
    # -----------------------------
    X = df.drop(columns=[target, "DIA"], errors="ignore")

    # -----------------------------
    # 7️⃣ Aplicar pipeline (importado desde tu archivo original)
    # -----------------------------
    pipeline = construir_pipeline(target, X)

    print("▶ Ajustando pipeline al dataset completo…")
    pipeline.fit(X, y)

    print("▶ Transformando dataset…")
    X_processed = pipeline.transform(X)

    print("✅ Shape final procesado:", X_processed.shape)

    return X_processed, dates, hours
    
def predict_consumption(filepath):
    """ Función principal de predicción """
    model = load_model() # Carga desde el registry
    X, dates, hours = load_and_preprocess(filepath)
    predictions = model.predict(X)

    return pd.DataFrame({
    'fecha': dates,
    'hora': hours,
    'prediccion_frio_kw': predictions
    })
 
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Uso: python predict.py <ruta_archivo>")
        sys.exit(1)

    filepath = sys.argv[1]

    results = predict_consumption(filepath)
    results.to_csv('results/predicciones.csv', index=False)
    print("Predicciones generadas en results/predicciones.csv")
