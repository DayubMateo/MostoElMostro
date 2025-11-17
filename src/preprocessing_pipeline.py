import pandas as pd
import numpy as np
import sys, os
import json
import datetime
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, PolynomialFeatures, OneHotEncoder, PowerTransformer
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
# Asumiendo que esta función existe en tu entorno
from auxiliar_functions import verificar_y_guardar_checksum
from sklearn.model_selection import train_test_split
from src.auxiliar_functions import importar_datos_completo

# --- CLASES DE TRANSFORMADORES ---

class RatioFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        print("CREANDO RATIOS!!1!")
        X = X.copy()
        X["ratio_SalaMaq_Planta"] = X["Sala Maq (Kw)"] / X["Planta (Kw)"]
        X["ratio_Servicios_Planta"] = X["Servicios (Kw)"] / X["Planta (Kw)"]
        X["ratio_Produccion_Planta"] = X["Produccion (Hl)"] / X["Planta (Kw)"]
        X["ratio_SalaMaq_Produccion"] = X["Sala Maq (Kw)"] / X["Produccion (Hl)"]
        X["ratio_Servicios_Produccion"] = X["Servicios (Kw)"] / X["Produccion (Hl)"]
        X["ratio_Planta_Produccion"] = X["Planta (Kw)"] / X["Produccion (Hl)"]
        X["ratio_Cocina_Produccion"] = X["Cocina (Kw)"] / X["Produccion (Hl)"]
        X["ratio_Cocina_SalaMaq"] = X["Cocina (Kw)"] / X["Sala Maq (Kw)"]
        X["ratio_Cocina_Servicios"] = X["Cocina (Kw)"] / X["Servicios (Kw)"]
        X["ratio_Servicios_SalaMaq"] = X["Servicios (Kw)"] / X["Sala Maq (Kw)"]
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        return X
    def set_output(self, *, transform=None):
        return self

class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, umbral=3.0):
        self.umbral = umbral
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        print("RECONOCIENDO OUTLIERS!!1!")
        X = X.copy()
        X_num = X.select_dtypes(include=[np.number])
        z_scores = np.abs(stats.zscore(X_num, nan_policy='omit'))
        z_scores_df = pd.DataFrame(z_scores, columns=X_num.columns, index=X.index)
        mask = z_scores_df > self.umbral
        X.loc[:, X_num.columns] = X_num.mask(mask, np.nan)
        return X
    def set_output(self, *, transform=None):
        return self

class PolynomialTopFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, top_n=15, grado=2):
        self.top_n = top_n
        self.grado = grado
        self.cols_poly = None
    def fit(self, X, y):
        print("POLYNOMIAL FEATURES!!1!")
        if y is None:
            raise ValueError("Se necesita y para calcular correlaciones con target")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        corr_spearman = X.corrwith(pd.Series(y, index=X.index), method='spearman')
        top = corr_spearman.abs().sort_values(ascending=False).head(self.top_n)
        self.cols_poly = top.index.tolist()
        return self
    def transform(self, X):
        if self.cols_poly is None:
            raise RuntimeError("Debe hacer fit antes de transform")
        X = X.copy()
        poly = PolynomialFeatures(degree=self.grado, include_bias=False)
        X_poly = pd.DataFrame(
            poly.fit_transform(X[self.cols_poly]),
            columns=poly.get_feature_names_out(self.cols_poly),
            index=X.index
        )
        nuevas = [c for c in X_poly.columns if c not in X.columns]
        X = pd.concat([X, X_poly[nuevas]], axis=1)
        return X
    def set_output(self, *, transform=None):
        return self

class TopNRandomForest(BaseEstimator, TransformerMixin):
    def __init__(self, n=50, random_state=42):
        self.n = n
        self.random_state = random_state
    def fit(self, X, y):
        print("FEATURE IMPORTANCES CON RANDOM FOREST!!1!")
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns
        else: 
            self.feature_names_ = [f"f{i}" for i in range(X.shape[1])]
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        self.rf_ = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=self.random_state)
        self.rf_.fit(X_scaled, y)
        importances = pd.Series(self.rf_.feature_importances_, index=self.feature_names_)
        self.top_features_ = importances.sort_values(ascending=False).head(self.n).index.tolist()
        self.top_indices_ = [self.feature_names_.get_loc(f) if hasattr(self.feature_names_, "get_loc") else i 
                             for i, f in enumerate(self.feature_names_) if f in self.top_features_]
        return self
    def transform(self, X):
        return X[:, self.top_indices_] if isinstance(X, np.ndarray) else X[self.top_features_]
    def set_output(self, *, transform=None):
        return self

class TopNLasso(BaseEstimator, TransformerMixin):
    def __init__(self, n=30, alpha=0.01, random_state=42):
        self.n = n
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y):
        print("FEATURE IMPORTANCES CON LASSO!!1!")
        
        # Convertir a DataFrame sí o sí
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

        self.feature_names_in_ = X.columns.tolist()

        # Escalado
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # LASSO
        self.lasso_ = Lasso(alpha=self.alpha, random_state=self.random_state)
        self.lasso_.fit(X_scaled, y)

        # top N
        coefs = np.abs(self.lasso_.coef_)
        self.top_indices_ = np.argsort(coefs)[-self.n:][::-1]
        self.top_features_ = [self.feature_names_in_[i] for i in self.top_indices_]

        return self

    def transform(self, X):
        # Convertir sí o sí a DataFrame y mantener nombres
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        return X[self.top_features_]
    
    def set_output(self, *, transform=None):
        return self

class CyclicalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.max_vals_ = None
        self.feature_names_in_ = None
    def fit(self, X, y=None):
        print("APRENDIENDO CICLOS (SIN/COS)!!1!")
        self.feature_names_in_ = X.columns.tolist()
        self.max_vals_ = X.max()
        return self
    def transform(self, X):
        X_transformed = pd.DataFrame(index=X.index)
        for col in self.feature_names_in_:
            max_val = self.max_vals_[col]
            if max_val == 0:
                X_transformed[f'{col}_sin'] = 0
                X_transformed[f'{col}_cos'] = 1 
            else:
                X_transformed[f'{col}_sin'] = np.sin(2 * np.pi * X[col] / max_val)
                X_transformed[f'{col}_cos'] = np.cos(2 * np.pi * X[col] / max_val)
        return X_transformed
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        output_features = []
        for col in input_features:
            output_features.append(f'{col}_sin')
            output_features.append(f'{col}_cos')
        return output_features
    def set_output(self, *, transform=None):
        return self

class ConstantFeatureRemover(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.keep_cols_ = None
        self.feature_names_in_ = None 
    
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            Xdf = X
        else:
            self.feature_names_in_ = [f"f{i}" for i in range(X.shape[1])]
            Xdf = pd.DataFrame(X, columns=self.feature_names_in_)
            
        self.keep_cols_ = Xdf.columns[Xdf.nunique() > 1].tolist()
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.keep_cols_] 
        
        Xdf = pd.DataFrame(X, columns=self.feature_names_in_)
        return Xdf[self.keep_cols_] 
    
    def set_output(self, *, transform=None):
        return self

class TimeSeriesInterpolatorSafe(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.last_train_row_ = None
        self.columns_ = None
    def fit(self, X, y=None):
        print("🚀 Interpolación temporal segura: FIT")
        X = pd.DataFrame(X).copy()
        self.columns_ = X.columns
        X = X.replace(0, np.nan)
        X_interp = (
            X
            .interpolate(method='linear', limit_direction='both')
            .ffill()
            .bfill()
        )
        self.last_train_row_ = X_interp.iloc[[-1]].copy()
        return self
    def transform(self, X):
        print("🔧 Interpolación temporal segura: TRANSFORM")
        original_index = X.index if isinstance(X, pd.DataFrame) else pd.RangeIndex(start=0, stop=len(X), step=1)
        X = pd.DataFrame(X).copy()
        X = X.replace(0, np.nan)
        prep = pd.concat([self.last_train_row_, X], ignore_index=True)
        prep_interp = (
            prep
            .interpolate(method='linear', limit_direction='forward')
            .ffill()
        )
        prep_interp = prep_interp.iloc[1:].reset_index(drop=True)
        prep_interp.columns = self.columns_
        prep_interp.index = original_index
        return prep_interp 
    def set_output(self, *, transform=None):
        return self



# --- FUNCIÓN DE PIPELINE CORREGIDA ---
def construir_pipeline(target, X, only_preprocess = False):
    print("DETECTANDO COLUMNAS!!1!")
    
    cyclical_cols = ['DIA_DEL_ANIO']
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    numeric_cols = [col for col in numeric_cols if col not in cyclical_cols and col != "es_finde"]
    
    power_inst = PowerTransformer(method='yeo-johnson')
    power_inst.set_output(transform="pandas")

    scaler_inst = StandardScaler()
    scaler_inst.set_output(transform="pandas")
    
    onehot_inst = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    onehot_inst.set_output(transform="pandas")
    
    lasso_selector = TopNLasso(n=40)
    lasso_selector.set_output(transform="pandas")

    numeric_transformer = Pipeline([
        ('outliers', OutlierReplacer(umbral=3.0)),
        ('imputer', TimeSeriesInterpolatorSafe()),
#       ('ratio', RatioFeatures()),
#       ('poly', PolynomialTopFeatures(top_n=15, grado=2)),
        ('const_drop', ConstantFeatureRemover()), 
        ('power', power_inst),                    
        ('scaler', scaler_inst)                   
    ])
    
    categorical_transformer = Pipeline([
        ('onehot', onehot_inst) 
    ])
    
    cyclical_transformer = Pipeline([
        ('cyclical', CyclicalEncoder()) 
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols),
        ('cyc', cyclical_transformer, cyclical_cols) 
    ], remainder='passthrough')
    
    preprocessor.set_output(transform="pandas")
    
    if only_preprocess:
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
    #        ('lasso_top40', lasso_selector) 
        ])
    else:
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('lasso_top40', lasso_selector) 
        ])
    
    return full_pipeline



# ---------- PREPROCESAMIENTO ----------
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
    importar_datos_completo()

    folder = 'data/processed'
    filename = 'dataset_final.csv'
    ruta_input = os.path.join(folder, filename)
    
    df = pd.read_csv(ruta_input, sep=',', decimal='.')
    df = df.sort_values(by='DIA', ignore_index=True)
    print(f"Shape original: {df.shape}")

    try:
        print("Convirtiendo 'DIA' a datetime y extrayendo 'DIA_DEL_ANIO'")
        df['DIA'] = pd.to_datetime(df['DIA'])
        df['DIA_DEL_ANIO'] = df['DIA'].dt.dayofyear
    except Exception as e:
        print(f"Error al convertir 'DIA'. Asegúrate de que sea un formato de fecha. Error: {e}")
        print("Continuando sin 'DIA_DEL_ANIO'. Es posible que el pipeline falle.")

    target = 'Frio (Kw) tomorrow'
    y = df[target]
    
    X = df.drop(columns=[target, 'DIA'], errors='ignore')
    
    test_size = 0.3
    split_index = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_index]
    test_df  = df.iloc[split_index:]

    X_train = train_df.drop(columns=['Frio (Kw) tomorrow', 'DIA'], errors='ignore')
    y_train = train_df['Frio (Kw) tomorrow']

    X_test = test_df.drop(columns=['Frio (Kw) tomorrow', 'DIA'], errors='ignore')
    y_test = test_df['Frio (Kw) tomorrow']

    pipeline = construir_pipeline(target, X)
    
    print("\n--- 🚀 Iniciando ajuste del Pipeline ---")
    pipeline.fit(X_train, y_train)
    print("--- ✅ Pipeline ajustado ---")

    X_train_preproc = pipeline.transform(X_train)
    X_test_preproc  = pipeline.transform(X_test)
    print("✅ Shape final de X_train preprocesada:", X_train_preproc.shape)
    print("✅ Shape final de X_test preprocesada:", X_test_preproc.shape)

    train_shape = X_train_preproc.shape
    test_shape = X_test_preproc.shape
    
    carpeta_salida = "data/processed"
    os.makedirs(carpeta_salida, exist_ok=True)

    ruta_X_train = os.path.join(carpeta_salida, "X_train_preproc.csv")
    X_train_preproc.to_csv(ruta_X_train, index=False)

    ruta_X_test = os.path.join(carpeta_salida, "X_test_preproc.csv")
    X_test_preproc.to_csv(ruta_X_test, index=False)

    ruta_y_train = os.path.join(carpeta_salida, "y_train.csv")
    y_train.to_csv(ruta_y_train, index=False)

    ruta_y_test = os.path.join(carpeta_salida, "y_test.csv")
    y_test.to_csv(ruta_y_test, index=False)

    print("✅ Todos los datasets guardados en data/processed:")     

    print("Almacenando pipeline...")

    # GUARDAR EL PIPELINE ENTRENADO
    print("Guardando pipeline entrenado...")
    ruta_pipeline = os.path.join('models', "pipeline_entrenado.joblib")
    joblib.dump(pipeline, ruta_pipeline)
    print(f"✅ Pipeline entrenado guardado en {ruta_pipeline}")

    print("Calculando checksum...")
    nombre_csv = "dataset_final.csv"
    ruta_csv = os.path.join(folder, nombre_csv) 
    verificar_y_guardar_checksum(ruta_csv)
    
    
    # --- BLOQUE DE DATA LINEAGE ---
    print("Generando data lineage...")
    try:
        pipeline_steps = [name for name, _ in pipeline.steps]
        
        final_features = []
        if hasattr(pipeline.named_steps['lasso_top40'], 'top_features_'):
            final_features = pipeline.named_steps['lasso_top40'].top_features_
        else:
            final_features = [f"col_{i}" for i in range(train_shape[1])] 
            
        lineage_info = {
            "input_file": ruta_input,
            "output_files": {
                "X_train": ruta_X_train,
                "X_test": ruta_X_test,
                "y_train": ruta_y_train,
                "y_test": ruta_y_test,
                "pipeline": ruta_pipeline,
            },
            "script_used": os.path.basename(__file__), 
            "execution_timestamp": datetime.datetime.now().isoformat(),
            "final_data_shape": {
                "X_train": {"rows": train_shape[0], "columns": train_shape[1]},
                "X_test": {"rows": test_shape[0], "columns": test_shape[1]},
                "y_train": {"rows": y_train.shape[0]},
                "y_test": {"rows": y_test.shape[0]}
            },
            "pipeline_summary": "Full preprocessing and feature selection pipeline.",
            "pipeline_steps": pipeline_steps,
            "final_features": final_features, 
            "description": "Dataset procesado con OutlierReplacer, TimeSeriesInterpolatorSafe, ConstantFeatureRemover, PowerTransformer, StandardScaler, CyclicalEncoder, y TopNLasso(n=40)."
        }
        
        ruta_lineage = os.path.join(carpeta_salida, "data_lineage.json")
        with open(ruta_lineage, 'w') as f:
            json.dump(lineage_info, f, indent=4)
        
        print(f"✅ Data lineage guardado en {ruta_lineage}")
    
    except Exception as e:
        print(f"❌ Error al generar el data lineage: {e}")