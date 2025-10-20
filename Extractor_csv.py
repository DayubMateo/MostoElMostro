import pandas as pd
import os
from collections import defaultdict

def unificar_hojas_de_excel(lista_de_archivos_excel, carpeta_salida):
    """
    Lee una lista de archivos Excel, une los datos de las hojas con el mismo nombre
    y guarda un archivo CSV consolidado por cada nombre de hoja.
    """
    # Crea la carpeta de salida si no existe
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)

    # 1. Usamos un diccionario para agrupar los DataFrames por nombre de hoja
    # Ej: {'Enero': [df_enero_2021, df_enero_2022], 'Febrero': [df_febrero_2021, ...]}
    datos_por_hoja = defaultdict(list)

    # 2. Recorremos cada archivo Excel de la lista
    for nombre_archivo in lista_de_archivos_excel:
        try:
            print(f"🔄 Procesando archivo: {nombre_archivo}...")
            # Leemos todas las hojas del excel actual
            diccionario_actual = pd.read_excel(nombre_archivo, sheet_name=None)
            
            # Agregamos cada hoja a nuestra estructura de datos principal
            for nombre_hoja, df in diccionario_actual.items():
                datos_por_hoja[nombre_hoja].append(df)
        
        except FileNotFoundError:
            print(f"❌ Error: No se encontró el archivo '{nombre_archivo}'")
        except Exception as e:
            print(f"❌ Ocurrió un error al leer '{nombre_archivo}': {e}")

    # 3. Ahora unimos (concatenamos) los datos y guardamos los CSV
    print("\n--- Unificando y guardando los datos ---")
    for nombre_hoja, lista_de_dataframes in datos_por_hoja.items():
        # pd.concat une una lista de DataFrames en uno solo
        df_final = pd.concat(lista_de_dataframes, ignore_index=True)
        
        nombre_archivo_csv = os.path.join(carpeta_salida, f"{nombre_hoja}.csv")
        
        df_final.to_csv(nombre_archivo_csv, index=False)
        print(f"✅ Hoja '{nombre_hoja}' guardada como '{nombre_archivo_csv}' con {len(df_final)} filas.")

# --- Configuración y Ejecución ---
nombres_archivos = ["Dataset_xlsx/Totalizadores Planta de Cerveza - 2022_2023.xlsx", 
                    "Dataset_xlsx/Totalizadores Planta de Cerveza 2021_2022.xlsx",
                    "Dataset_xlsx/Totalizadores Planta de Cerveza 2023_2024.xlsx"]
carpeta_salida = "Dataset_csv_unificado"

unificar_hojas_de_excel(nombres_archivos, carpeta_salida)

