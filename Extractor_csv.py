import pandas as pd
import os

def unir_todos_los_excels_en_un_csv(lista_de_archivos_excel, carpeta_salida, nombre_csv_salida):
    """
    Lee una lista de archivos Excel, itera sobre todas sus hojas,
    filtra las filas donde la columna 'HORA' sea '23:59:00',
    LUEGO agrupa por 'DIA' y suma las columnas numéricas,
    y guarda todo en un único archivo CSV.
    """
    print("Iniciando proceso de filtrado y unificación...")
    dataframes_filtrados = []

    # 1. Asegurar que la carpeta de salida exista
    try:
        os.makedirs(carpeta_salida, exist_ok=True)
        print(f"Carpeta de salida verificada: {carpeta_salida}")
    except Exception as e:
        print(f"FATAL: No se pudo crear la carpeta de salida. Motivo: {e}")
        return

    # 2. Iterar sobre cada archivo Excel en la lista
    for archivo_excel in lista_de_archivos_excel:
        print(f"\n--- Procesando archivo: {archivo_excel} ---")
        try:
            todas_las_hojas = pd.read_excel(archivo_excel, sheet_name=None)
        except Exception as e:
            print(f"  ERROR: No se pudo leer el archivo. Motivo: {e}. Omitiendo este archivo.")
            continue

        # 3. Iterar sobre cada hoja leída
        for nombre_hoja, df_hoja in todas_las_hojas.items():
            print(f"  -> Procesando hoja: '{nombre_hoja}'")

            # 4. Verificar si existe la columna 'HORA'
            if 'HORA' not in df_hoja.columns:
                print(f"     ADVERTENCIA: La hoja '{nombre_hoja}' no tiene columna 'HORA'. Omitiendo esta hoja.")
                continue
                
            # --- Adicional: Verificar si existe 'DIA' ---
            # Si no hay 'DIA', no podremos agrupar al final.
            if 'DIA' not in df_hoja.columns:
                print(f"     ADVERTENCIA: La hoja '{nombre_hoja}' no tiene columna 'DIA'. Omitiendo esta hoja.")
                continue

            try:
                # 5. EL FILTRADO
                filtro_hora = df_hoja['HORA'].astype(str).str.strip().str.endswith("23:59:00").fillna(False)
                df_filtrado = df_hoja[filtro_hora]

                # 6. Guardar los resultados
                if not df_filtrado.empty:
                    print(f"     Se encontraron {len(df_filtrado)} filas con HORA='23:59:00'.")
                    dataframes_filtrados.append(df_filtrado)
                else:
                    print("     No se encontraron filas que coincidan.")
            
            except Exception as e:
                print(f"     ERROR al filtrar la hoja '{nombre_hoja}'. Motivo: {e}")

    # 7. Combinar todos los DataFrames si se encontró algo
    if not dataframes_filtrados:
        print("\nADVERTENCIA: No se encontró ningún dato en ningún archivo con el filtro especificado.")
        print("No se generará ningún archivo CSV.")
        return

    print(f"\nCombinando {len(dataframes_filtrados)} bloques de datos encontrados...")
    df_final = pd.concat(dataframes_filtrados, ignore_index=True)
    print(f"Se combinaron un total de {len(df_final)} filas filtradas.")

    # --- PASO 7.5: NUEVA AGREGACIÓN POR DÍA ---
    print("\nIniciando agregación por 'DIA'...")

    # Verificar si 'DIA' existe en el DF combinado
    if 'DIA' not in df_final.columns:
        print("FATAL: La columna 'DIA' no se encontró en los datos combinados. No se puede agrupar.")
        return
        
    try:
        # Agrupar por 'DIA', sumar solo columnas numéricas.
        # .sum() por defecto trata los NaNs como 0 (skipna=True)
        # as_index=False mantiene 'DIA' como una columna.
        df_agregado = df_final.groupby('DIA', as_index=False).sum(numeric_only=True)
        
        print(f"Se agruparon los datos en {len(df_agregado)} filas (una por cada día único).")

    except Exception as e:
        print(f"ERROR: No se pudo completar la agregación por 'DIA'. Motivo: {e}")
        return

    # 8. Guardar en el CSV final
    ruta_salida_completa = os.path.join(carpeta_salida, nombre_csv_salida)
    
    try:
        # Usamos 'utf-8-sig' para asegurar compatibilidad con Excel
        # IMPORTANTE: Guardamos df_agregado, no df_final
        df_agregado.to_csv(ruta_salida_completa, index=False, encoding='utf-8-sig')
        print(f"\n¡PROCESO COMPLETADO! 🚀")
        print(f"Se guardaron {len(df_agregado)} filas agregadas en total.")
        print(f"Archivo final: {ruta_salida_completa}")
    except Exception as e:
        print(f"\nERROR: No se pudo guardar el archivo CSV final. Motivo: {e}")

# --- EJECUCIÓN DEL CÓDIGO ---
# (Usando las variables que me pasaste)

# 1. Define la lista de archivos a procesar
nombres_archivos_ordenados = [
    "Dataset_xlsx/Totalizadores Planta 2022_2023.xlsx",
    "Dataset_xlsx/Totalizadores Planta - 2021_2023.xlsx",
    "Dataset_xlsx/Totalizadores Planta 2020_2022.xlsx"
]

# 2. Define la carpeta de salida y el nombre del archivo
carpeta_salida = "Dataset_csv"
nombre_csv_salida = "Dataset.csv"

# 3. Llama a la función
unir_todos_los_excels_en_un_csv(nombres_archivos_ordenados, carpeta_salida, nombre_csv_salida)