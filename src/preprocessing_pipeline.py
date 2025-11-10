import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, PolynomialFeatures, OneHotEncoder, PowerTransformer
from sklearn.impute import KNNImputer
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
from auxiliar_functions import verificar_y_guardar_checksum
from sklearn.model_selection import train_test_split

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


# ---- Transformer de selección top N por Lasso ----
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



# ---------- Función de construcción del pipeline ----------
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.impute import KNNImputer

def construir_pipeline(target, X):
    # Detectar columnas
    print("DETECTANDO COLUMNAS!!1!")
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Pipeline para columnas numéricas
    numeric_transformer = Pipeline([
        ('ratio', RatioFeatures()),
        ('poly', PolynomialTopFeatures(top_n=15, grado=2)),
        ('outliers', OutlierReplacer(umbral=3.0)),
        ('imputer', KNNImputer(n_neighbors=20)),
        ('power', PowerTransformer(method='yeo-johnson')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline para columnas categóricas
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
    ])
    
    # Preprocessor combinado
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='passthrough')
    
    # Pipeline completo: primero preprocessing, después selección de features
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('rf_top50', TopNRandomForest(n=50)),
        ('lasso_top30', TopNLasso(n=30))
    ])
    
    return full_pipeline



# ---------- Ejemplo de uso ----------
if __name__ == "__main__":
    # Cargar datos
    folder = 'data/processed'
    filename = 'dataset_final.csv'
    df = pd.read_csv(f'{folder}/{filename}', sep=',', decimal='.')
    print(df.shape)
    df["Frio (Kw)_movil_5"] = df["Frio (Kw)"].rolling(window=5, min_periods=1).mean()
    df["finde"] = df["Dia_semana"].isin(["Sabado", "Domingo"]).astype(int)

    target = 'Frio (Kw)'
    y = df[target]
    X = df.drop(columns=[target, 'DIA'], errors='ignore')
    
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True
    )

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
    pd.DataFrame(X_train_preproc, index=X_train.index).to_csv(ruta_X_train, index=False)

    # Guardar X_test preprocesado
    ruta_X_test = os.path.join(carpeta_salida, "X_test_preproc.csv")
    pd.DataFrame(X_test_preproc, index=X_test.index).to_csv(ruta_X_test, index=False)

    # Guardar y_train
    ruta_y_train = os.path.join(carpeta_salida, "y_train.csv")
    y_train.to_csv(ruta_y_train, index=False)

    # Guardar y_test
    ruta_y_test = os.path.join(carpeta_salida, "y_test.csv")
    y_test.to_csv(ruta_y_test, index=False)

    print("✅ Todos los datasets guardados en data/processed:")    

    # Calcular Checksum
    carpeta_salida = "data/processed"
    nombre_csv = "dataset_final.csv"
    ruta_csv = os.path.join(carpeta_salida, nombre_csv)  # ruta completa al archivo
    verificar_y_guardar_checksum(ruta_csv)

