import pandas as pd
import numpy as np
import sys, os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, PolynomialFeatures, OneHotEncoder, PowerTransformer
from sklearn.impute import KNNImputer
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
# Asumiendo que esta función existe en tu entorno
from auxiliar_functions import verificar_y_guardar_checksum
from sklearn.model_selection import train_test_split
from src.auxiliar_functions import importar_datos_completo

# ---------- Transformador personalizado para ratios ----------
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

# ---------- Transformador personalizado para outliers ----------
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

# ---------- Transformador para polinomios sobre top N correlaciones ----------
class PolynomialTopFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, top_n=15, grado=2):
        self.top_n = top_n
        self.grado = grado
        self.cols_poly = None
    
    def fit(self, X, y):
        """
        X: DataFrame numérico
        y: target (Series o array-like)
        """
        print("POLYNOMIAL FEATURES!!1!")
        if y is None:
            raise ValueError("Se necesita y para calcular correlaciones con target")
        
        # Asegurarnos de tener DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Correlación Spearman con y
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
        # Agregar solo columnas nuevas
        nuevas = [c for c in X_poly.columns if c not in X.columns]
        X = pd.concat([X, X_poly[nuevas]], axis=1)
        return X


# ---- Transformer de selección top N por Random Forest ----
class TopNRandomForest(BaseEstimator, TransformerMixin):
    def __init__(self, n=50, random_state=42):
        self.n = n
        self.random_state = random_state
    
    def fit(self, X, y):
        # Guardar nombres de columnas
        print("FEATURE IMPORTANCES CON RANDOM FOREST!!1!")
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns
        else:  # si es ndarray
            self.feature_names_ = [f"f{i}" for i in range(X.shape[1])]
        
        # Escalamos
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Random Forest
        self.rf_ = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=self.random_state)
        self.rf_.fit(X_scaled, y)
        
        # Guardamos top N features
        importances = pd.Series(self.rf_.feature_importances_, index=self.feature_names_)
        self.top_features_ = importances.sort_values(ascending=False).head(self.n).index.tolist()
        # Guardamos los índices
        self.top_indices_ = [self.feature_names_.get_loc(f) if hasattr(self.feature_names_, "get_loc") else i 
                             for i, f in enumerate(self.feature_names_) if f in self.top_features_]
        return self
    
    def transform(self, X):
        # Seleccionar columnas por índice
        return X[:, self.top_indices_] if isinstance(X, np.ndarray) else X[self.top_features_]


# ---- Transformer de selección top N por Lasso (compatible con ndarray) ----
class TopNLasso(BaseEstimator, TransformerMixin):
    def __init__(self, n=30, alpha=0.01, random_state=42):
        self.n = n
        self.alpha = alpha
        self.random_state = random_state
    
    def fit(self, X, y):
        # Escalado
        print("FEATURE IMPORTANCES CON LASSO!!1!")
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Ajustar Lasso
        self.lasso_ = Lasso(alpha=self.alpha, random_state=self.random_state)
        self.lasso_.fit(X_scaled, y)
        
        # Guardar índices de las top N features
        coefs = np.abs(self.lasso_.coef_)
        self.top_indices_ = np.argsort(coefs)[-self.n:][::-1]  # top N índices, descendente
        return self
    
    def transform(self, X):
        # Seleccionar columnas por índice
        return X[:, self.top_indices_] if isinstance(X, np.ndarray) else X.iloc[:, self.top_indices_]


# ---------- [NUEVO] Transformador para Seno/Coseno ----------
class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """
    Codifica features numéricas cíclicas (ej. día del año, mes) en sin y cos.
    Espera un DataFrame en fit y transform.
    """
    def __init__(self):
        self.max_vals_ = None
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        print("APRENDIENDO CICLOS (SIN/COS)!!1!")
        self.feature_names_in_ = X.columns.tolist()
        # Aprende el valor máximo (el periodo) de cada columna cíclica
        self.max_vals_ = X.max()
        return self

    def transform(self, X):
        X_transformed = pd.DataFrame(index=X.index)
        for col in self.feature_names_in_:
            max_val = self.max_vals_[col]
            if max_val == 0:
                X_transformed[f'{col}_sin'] = 0
                X_transformed[f'{col}_cos'] = 1 # cos(0)
            else:
                # Calculamos sin y cos basados en el periodo (max_val)
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


class ConstantFeatureRemover(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.keep_cols_ = None
    
    def fit(self, X, y=None):
        Xdf = pd.DataFrame(X)
        self.keep_cols_ = Xdf.columns[Xdf.nunique() > 1].tolist()
        return self
    
    def transform(self, X):
        Xdf = pd.DataFrame(X)
        return Xdf[self.keep_cols_].values


# ==============================
#  TRANSFORMER: Interpolación temporal segura
# ==============================
class TimeSeriesInterpolatorSafe(BaseEstimator, TransformerMixin):
    """
    Interpolación temporal segura:
      - En fit(): imputación SOLO usando datos del train (sin leakage).
      - Usa: replace(0->nan), interpolación linear + ffill + bfill.
      - En transform(): el test se interpola usando únicamente la última
        fila imputada del train como semilla (forward-fill).
    """

    def __init__(self):
        self.last_train_row_ = None
        self.columns_ = None

    def fit(self, X, y=None):
        print("🚀 Interpolación temporal segura: FIT")

        # Asegurar DataFrame y copiar
        X = pd.DataFrame(X).copy()
        self.columns_ = X.columns

        # ---- 1) Reemplazar 0 → NaN ----
        X = X.replace(0, np.nan)

        # ---- 2) Interpolación SOLO en Train ----
        X_interp = (
            X
            .interpolate(method='linear', limit_direction='both')
            .ffill()
            .bfill()
        )

        # ---- 3) Guardar última fila imputada ----
        self.last_train_row_ = X_interp.iloc[[-1]].copy()

        return self

    def transform(self, X):
        print("🔧 Interpolación temporal segura: TRANSFORM")

        X = pd.DataFrame(X).copy()

        # ---- 1) Reemplazar 0 → NaN ----
        X = X.replace(0, np.nan)

        # ---- 2) Concatenar semilla + X_test ----
        prep = pd.concat([self.last_train_row_, X], ignore_index=True)

        # ---- 3) Interpolación forward ONLY (no leakage) ----
        prep_interp = (
            prep
            .interpolate(method='linear', limit_direction='forward')
            .ffill()
        )

        # ---- 4) Quitar semilla ----
        prep_interp = prep_interp.iloc[1:].reset_index(drop=True)

        # ---- 5) Restaurar columnas originales ----
        prep_interp.columns = self.columns_

        return prep_interp



# ---------- [MODIFICADO] Función de construcción del pipeline ----------
def construir_pipeline(target, X):
    # Detectar columnas
    print("DETECTANDO COLUMNAS!!1!")
    
    # --- AÑADIDO ---
    # Definimos explícitamente qué columna es cíclica
    # Usamos 'DIA_DEL_ANIO' que crearemos en el main
    cyclical_cols = ['DIA_DEL_ANIO']
    
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # --- MODIFICADO ---
    # Excluimos la columna cíclica de las numéricas para darle tratamiento especial
    numeric_cols = [col for col in numeric_cols if col not in cyclical_cols and col != "es_finde"]
    
    # Pipeline para columnas numéricas
    numeric_transformer = Pipeline([
        ('outliers', OutlierReplacer(umbral=3.0)),
        ('imputer', TimeSeriesInterpolatorSafe()),
#        ('ratio', RatioFeatures()),
#        ('poly', PolynomialTopFeatures(top_n=15, grado=2)),
        ('const_drop', ConstantFeatureRemover()),
        ('power', PowerTransformer(method='yeo-johnson')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline para columnas categóricas
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    # --- AÑADIDO ---
    # Pipeline para columnas cíclicas (seno/coseno)
    cyclical_transformer = Pipeline([
        ('cyclical', CyclicalEncoder())
        # No necesita escalado, sin/cos ya están en [-1, 1]
    ])
    
    # Preprocessor combinado
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols),
        ('cyc', cyclical_transformer, cyclical_cols) 
    ], remainder='passthrough')
    
    # Pipeline completo: primero preprocessing, después selección de features
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('lasso_top40', TopNLasso(n=40))
    ])
    
    return full_pipeline



# ---------- PREPROCESAMIENTO ----------
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
    importar_datos_completo()

    # Cargar datos
    folder = 'data/processed'
    filename = 'dataset_final.csv'
    df = pd.read_csv(f'{folder}/{filename}', sep=',', decimal='.')
    df = df.sort_values(by='DIA', ignore_index=True)
    print(f"Shape original: {df.shape}")

    # --- AÑADIDO: Procesamiento de 'DIA' ---
    # Asumimos que 'DIA' es una columna tipo 'YYYY-MM-DD' o similar
    try:
        print("Convirtiendo 'DIA' a datetime y extrayendo 'DIA_DEL_ANIO'")
        df['DIA'] = pd.to_datetime(df['DIA'])
        # Creamos 'DIA_DEL_ANIO' (ej: 1 a 365). Esta es la que usará el pipeline
        df['DIA_DEL_ANIO'] = df['DIA'].dt.dayofyear
    except Exception as e:
        print(f"Error al convertir 'DIA'. Asegúrate de que sea un formato de fecha. Error: {e}")
        print("Continuando sin 'DIA_DEL_ANIO'. Es posible que el pipeline falle.")

    df["Frio (Kw)_movil_5"] = df["Frio (Kw)"].rolling(window=5, min_periods=1).mean()
    df["finde"] = df["Dia_semana"].isin(["Sabado", "Domingo"]).astype(int)

    target = 'Frio (Kw) tomorrow'
    y = df[target]
    
    # --- MODIFICADO ---
    # Dropeamos el target y la columna 'DIA' original (que ya es datetime)
    # MANTENEMOS 'DIA_DEL_ANIO' que creamos
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
    
    # Ajustamos
    pipeline.fit(X_train, y_train)

    # Transformamos
    X_train_preproc = pipeline.transform(X_train)
    X_test_preproc  = pipeline.transform(X_test)
    print("✅ Shape final de X_train preprocesada:", X_train_preproc.shape)
    print("✅ Shape final de X_test preprocesada:", X_test_preproc.shape)

    carpeta_salida = "data/processed"
    os.makedirs(carpeta_salida, exist_ok=True)

    # Guardar X_train preprocesado
    ruta_X_train = os.path.join(carpeta_salida, "X_train_preproc.csv")
    # Es importante guardar como array, ya que los nombres de columna se pierden
    pd.DataFrame(X_train_preproc).to_csv(ruta_X_train, index=False)

    # Guardar X_test preprocesado
    ruta_X_test = os.path.join(carpeta_salida, "X_test_preproc.csv")
    pd.DataFrame(X_test_preproc).to_csv(ruta_X_test, index=False)

    # Guardar y_train
    ruta_y_train = os.path.join(carpeta_salida, "y_train.csv")
    y_train.to_csv(ruta_y_train, index=False)

    # Guardar y_test
    ruta_y_test = os.path.join(carpeta_salida, "y_test.csv")
    y_test.to_csv(ruta_y_test, index=False)

    print("✅ Todos los datasets guardados en data/processed:")    

    # Calcular Checksum (comentado si 'auxiliar_functions' no está)
    print("Calculando checksum...")
    nombre_csv = "dataset_final.csv"
    ruta_csv = os.path.join(folder, nombre_csv) # ruta al archivo original
    verificar_y_guardar_checksum(ruta_csv)