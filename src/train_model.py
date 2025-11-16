import os
import json
import pandas as pd
import joblib
from datetime import datetime
import subprocess
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import ElasticNet

# ======================================================
# 1. Cargar experiment_logs.csv y elegir el mejor modelo
# ======================================================
def load_best_model_entry(log_path="notebooks/results/experiment_logs.csv"):
    # Si no existe el archivo, crearlo vacío con headers y salir
    if not os.path.exists(log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        df_empty = pd.DataFrame(columns=[
            "timestamp","model_name","optuna_cv_mae",
            "mae_train","mae_test","rmse_train","rmse_test",
            "r2_train","r2_test","hyperparameters"
        ])
        df_empty.to_csv(log_path, index=False)

        print(f"⚠️ No existía {log_path}. Se creó uno vacío.")
        print("👉 Corré primero tus experimentos Optuna para generar resultados.")
        exit(0)

    df = pd.read_csv(
        log_path,
        quotechar='"',
        escapechar='\\',
        engine="python"   # más tolerante
    )

    if df.empty:
        print(f"⚠️ {log_path} existe pero está vacío.")
        print("👉 Corré tus experimentos Optuna antes de entrenar el modelo final.")
        exit(0)

    df_sorted = df.sort_values("mae_test", ascending=True)
    best = df_sorted.iloc[0]
    return best



# ======================================================
# 2. Mapear nombres a clases reales de modelos
# ======================================================
MODEL_MAP = {
    "RandomForest": RandomForestRegressor,
    "XGBoost": xgb.XGBRegressor,
    "LightGBM": lgb.LGBMRegressor,
    "Pipeline_ElasticNet": ElasticNet
}


# ======================================================
# 3. Generar versión semántica automáticamente
# ======================================================
def get_next_version(registry_path="models/model_registry.json"):

    if not os.path.exists(registry_path):
        return "v1.0.0"

    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    last_version = registry[-1]["version"]

    major, minor, patch = map(int, last_version.replace("v", "").split("."))

    # Estrategia simple → sube el patch para cada nuevo modelo
    patch += 1

    return f"v{major}.{minor}.{patch}"


# ======================================================
# 4. Obtener hash del commit actual de git
# ======================================================
def get_git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.STDOUT
        ).decode("utf-8").strip()
    except:
        return "NO_GIT_REPO"


# ======================================================
# 5. Entrenar modelo final y guardarlo
# ======================================================
def train_final_model():
    # --- Cargar mejor modelo del log ---
    best = load_best_model_entry()

    model_name = best["model_name"]
    params = json.loads(best["hyperparameters"])
    mae_test = float(best["mae_test"])

    ModelClass = MODEL_MAP[model_name]

    # Muchos modelos necesitan random_state
    if model_name in ["RandomForest", "Pipeline_ElasticNet"]:
        params["random_state"] = 42
    if model_name == "RandomForest":
        params["n_jobs"] = -1
    if model_name == "XGBoost":
        params["verbosity"] = 0
    if model_name == "LightGBM":
        params["verbose"] = -1

    model = ModelClass(**params)

    # --- Cargar datasets ---
    X_train = pd.read_csv("data/processed/X_train_preproc.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

    X_test = pd.read_csv("data/processed/X_test_preproc.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    # --- Unir los datasets ---
    X_full = pd.concat([X_train, X_test], ignore_index=True)
    y_full = pd.concat([pd.Series(y_train), pd.Series(y_test)], ignore_index=True)

    # --- Entrenar modelo final ---
    print(f"\nEntrenando modelo final '{model_name}' con los mejores hiperparámetros...")
    model.fit(X_full, y_full)

    # --- Versión del modelo ---
    version = get_next_version()

    os.makedirs("models", exist_ok=True)
    model_path = f"models/modelo_{version}.pkl"

    joblib.dump(model, model_path)
    print(f"✅ Modelo guardado en: {model_path}")

    # --- Registrar en model_registry.json ---
    registry_path = "models/model_registry.json"

    registry_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": version,
        "model_name": model_name,
        "hyperparameters": params,
        "mae_test": mae_test,
        "model_path": model_path,
        "git_commit": get_git_commit_hash()
    }

    if not os.path.exists(registry_path):
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump([registry_entry], f, indent=4)
    else:
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
        registry.append(registry_entry)
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=4)

    print("📄 Entrada registrada en models/model_registry.json")



# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    train_final_model()
    print("\n🎉 Entrenamiento final completado con éxito.\n")
