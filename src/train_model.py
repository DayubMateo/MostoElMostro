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


def normalize_params(params):
    """
    Normaliza un diccionario de hiperparámetros para que:
    - El orden no afecte
    - Los tipos numpy se conviertan a tipos nativos
    """
    normalized = {}

    for k, v in params.items():
        # Convertir tipos numpy a python nativo
        if hasattr(v, "item"):
            v = v.item()

        # Convertir listas numpy a listas normales
        if isinstance(v, (list, tuple)):
            v = [x.item() if hasattr(x, "item") else x for x in v]

        normalized[k] = v

    # devolver versión ordenada
    return dict(sorted(normalized.items()))


def get_next_version(best, registry_path="models/model_registry.json"):
    if not os.path.exists(registry_path):
        return "v1.0.0"

    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    last = registry[-1]
    last_version = last["version"]
    last_model = last["model_name"]

    last_params = normalize_params(last["hyperparameters"])
    current_params = normalize_params(json.loads(best["hyperparameters"]))

    current_model = best["model_name"]

    major, minor, patch = map(int, last_version.replace("v", "").split("."))

    # Regla 1
    if current_model != last_model:
        return f"v{major+1}.0.0"

    # Regla 2
    if current_params != last_params:
        return f"v{major}.{minor+1}.0"

    # Regla 3
    return f"v{major}.{minor}.{patch+1}"



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

    # ================================
    #  VERSIONAMIENTO SEMÁNTICO NUEVO
    # ================================
    # Normalizar los hiperparámetros actuales igual que quedarán en el registro
    params_normalized = normalize_params(params)

    # También normalizamos los hiperparámetros del mejor modelo del log
    best["hyperparameters"] = json.dumps(params_normalized)
    version = get_next_version(best)

    # ================================
    #  ELIMINAR MODELO ANTERIOR
    # ================================
    registry_path = "models/model_registry.json"
    if os.path.exists(registry_path):
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)

        last_entry = registry[-1]
        old_model_path = last_entry.get("model_path", None)

        if old_model_path and os.path.exists(old_model_path):
            try:
                os.remove(old_model_path)
                print(f"🗑️ Modelo anterior eliminado: {old_model_path}")
            except Exception as e:
                print(f"⚠️ No pude eliminar {old_model_path}: {e}")

    # --- Guardar nuevo modelo ---
    os.makedirs("models", exist_ok=True)
    model_path = f"models/modelo_{version}.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Modelo guardado en: {model_path}")

    # --- Registrar ---
    registry_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": version,
        "model_name": model_name,
        "hyperparameters": params_normalized,
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
