import pandas as pd
import os

def unir_todos_los_excels_en_un_csv(lista_de_archivos_excel, carpeta_salida, nombre_csv_salida, columna_fecha):
    """
    Une TODAS las hojas de varios archivos Excel en un único CSV,
    manteniendo todas las columnas y sin omitir ninguna fila.
    Las columnas que no existan en algunos datasets se completan con NaN.
    """
    
    # Crear carpeta de salida si no existe
    os.makedirs(carpeta_salida, exist_ok=True)
    
    # Lista para almacenar todos los DataFrames combinados
    datos_combinados = []
    
    # Procesar cada archivo en el orden especificado
    for archivo in lista_de_archivos_excel:
        print(f"Procesando archivo: {archivo}")
        
        # Leer todas las hojas del archivo Excel
        hojas = pd.read_excel(archivo, sheet_name=None)
        
        # Combinar todas las hojas de este archivo
        datos_archivo = []
        for nombre_hoja, df_hoja in hojas.items():
            datos_archivo.append(df_hoja)
        
        # Concatenar todas las hojas del archivo actual
        if datos_archivo:
            df_archivo_completo = pd.concat(datos_archivo, ignore_index=True, sort=False)
            datos_combinados.append(df_archivo_completo)
    
    # Combinar todos los archivos en el orden especificado
    if datos_combinados:
        # El primer DataFrame (más reciente/informativo) es la base
        df_final = datos_combinados[0]
        
        # Añadir los DataFrames restantes
        for i in range(1, len(datos_combinados)):
            df_final = pd.concat([df_final, datos_combinados[i]], ignore_index=True, sort=False)
        
        # Ruta completa del archivo de salida
        ruta_salida = os.path.join(carpeta_salida, nombre_csv_salida)
        
        # Guardar como CSV
        df_final.to_csv(ruta_salida, index=False, encoding='utf-8')
        print(f"✅ CSV guardado en: {ruta_salida}")
        print(f"📊 Dimensiones del dataset final: {df_final.shape}")
        print(f"📅 Columna de fecha: '{columna_fecha}'")
        
        # Mostrar información sobre columnas
        print("\n🔍 Columnas en el dataset final:")
        for columna in df_final.columns:
            print(f"   - {columna}")
            
    else:
        print("❌ No se encontraron datos para combinar.")

# --- Configuración y ejecución ---
NOMBRE_COLUMNA_FECHA = "DIA"

nombres_archivos_ordenados = [
    "Dataset_xlsx/Totalizadores Planta 2022_2023.xlsx",
    "Dataset_xlsx/Totalizadores Planta - 2021_2023.xlsx", 
    "Dataset_xlsx/Totalizadores Planta 2020_2022.xlsx"
]

carpeta_salida = "Dataset_csv_unificado_completo"
nombre_csv_salida = "Totalizadores_TODO.csv"

unir_todos_los_excels_en_un_csv(nombres_archivos_ordenados, carpeta_salida, nombre_csv_salida, NOMBRE_COLUMNA_FECHA)
