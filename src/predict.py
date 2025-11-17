# Estructura requerida del script
import pandas as pd
import joblib
import sys, re
import json
import os
from preprocessing_pipeline import construir_pipeline
from src.auxiliar_functions import importar_datos_completo
from preprocessing_pipeline import PolynomialTopFeatures, RatioFeatures, OutlierReplacer, ConstantFeatureRemover, TimeSeriesInterpolatorSafe, CyclicalEncoder, TopNLasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder, PowerTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime


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


def generar_nombre_csv(filepath):
    """
    Extrae años del nombre del archivo y genera un nombre tipo dataset_2023-2024.csv
    """
    filename = os.path.basename(filepath)

    # Buscar dos años consecutivos AAAA_AAAA dentro del nombre
    match = re.search(r'(\d{4})[^\d]?(\d{4})', filename)

    if not match:
        raise ValueError(f"No se pudieron detectar años en el archivo: {filename}")

    year1, year2 = match.groups()

    # Generar nombre limpio
    return f"dataset_{year1}-{year2}.csv"

def load_and_preprocess(filepath):
    nombre_archivo = generar_nombre_csv(filepath)

    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
    importar_datos_completo(archivos = [filepath], nombre_csv = nombre_archivo)

    csv_path = os.path.join("data", "processed", nombre_archivo)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No existe el CSV procesado: {csv_path}")

    df = pd.read_csv(csv_path, sep=",", decimal=".")

    try:
        print("Convirtiendo 'DIA' a datetime y extrayendo 'DIA_DEL_ANIO'")
        df['DIA'] = pd.to_datetime(df['DIA'])
        df['DIA_DEL_ANIO'] = df['DIA'].dt.dayofyear
    except Exception as e:
        print(f"Error al convertir 'DIA'. Asegúrate de que sea un formato de fecha. Error: {e}")
        print("Continuando sin 'DIA_DEL_ANIO'. Es posible que el pipeline falle.")

    target = "Frio (Kw) tomorrow"
    y = df[target]

    dates = df["DIA"] if "DIA" in df.columns else None
    hours = df["HORA"] if "HORA" in df.columns else None

    X = df.drop(columns=[target, 'DIA'], errors="ignore")

    # Cargar pipeline de preprocesamiento
    preprocessing_pipeline = joblib.load("models/pipeline_entrenado.joblib")
    
    # Aplicar pipeline
    X = preprocessing_pipeline.transform(X)

    return X, dates, hours

    
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

def evaluate_with_training_data(predictions_df, filepath = "data/processed/dataset_2022-2023.csv"):
    """
    Evalúa las predicciones comparando con filepath para las mismas fechas
    """
    print("\n🔍 Buscando datos reales para evaluación...")
    
    # Cargar dataset con los valores reales
    try:
        df_real = pd.read_csv(filepath)
        df_real['DIA'] = pd.to_datetime(df_real['DIA'])
        
        # Crear DataFrame con valores reales
        real_df = pd.DataFrame({
            'fecha': df_real['DIA'],
            'y_real': df_real['Frio (Kw) tomorrow']
        })
        
    except Exception as e:
        print(f"⚠️  Error cargando dataset con valores reales: {e}")
        print(f"   Asegúrate de que existe {filepath}")
        return None, None
    
    # Preparar predictions_df para matching
    predictions_df['fecha'] = pd.to_datetime(predictions_df['fecha'])
    
    # Hacer el matching solo por fecha (día)
    merged_df = pd.merge(
        predictions_df, 
        real_df, 
        on=['fecha'], 
        how='inner',
        suffixes=('_pred', '_real')
    )
    
    if len(merged_df) == 0:
        print("❌ No se encontraron coincidencias entre las predicciones y los datos reales")
        print("   Fechas en predictions:", predictions_df['fecha'].min(), "a", predictions_df['fecha'].max())
        print("   Fechas en datos reales:", real_df['fecha'].min(), "a", real_df['fecha'].max())
        return None, None
    
    print(f"✅ Encontradas {len(merged_df)} coincidencias para evaluación")
    
    # Calcular métricas
    y_true = merged_df['y_real']
    y_pred = merged_df['prediccion_frio_kw']
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'samples_matched': len(merged_df),
        'total_predictions': len(predictions_df),
        'match_rate': len(merged_df) / len(predictions_df)
    }
    
    # Análisis detallado
    print(f"\n📊 Evaluación vs Datos Reales:")
    print(f"   MAE:  {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R²:   {r2:.4f}")
    print(f"   Muestras coincidentes: {len(merged_df)}/{len(predictions_df)} ({metrics['match_rate']:.1%})")
    
    # Mostrar algunas predicciones vs reales
    print(f"\n🔍 Ejemplos de comparaciones:")
    sample_comparisons = merged_df.head(5)
    for _, row in sample_comparisons.iterrows():
        error = row['y_real'] - row['prediccion_frio_kw']
        print(f"   {row['fecha'].strftime('%Y-%m-%d')} | "
              f"Real: {row['y_real']:.2f} | Pred: {row['prediccion_frio_kw']:.2f} | "
              f"Error: {error:.2f}")
    
    
    return metrics, merged_df
 
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Uso: python predict.py <ruta_archivo>")
        sys.exit(1)

    filepath = sys.argv[1]

    results = predict_consumption(filepath)

    # Modificar lo que viene delante de processed/ para que se adecue al csv generado en data/processed/dataset_año-año.csv.
    # Ejecutar si se desea visualizar la calidad de las predicciones.
    #evaluate_with_training_data(results, "data/processed/dataset_2022-2023.csv")
    # ------------
    results.to_csv('results/predicciones.csv', index=False)
    print("Predicciones generadas en results/predicciones.csv")


