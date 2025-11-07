# Si se realiza un cambio en este archivo,
# se debe reiniciar el visual para que se
# implementen los cambios.

COLUMNAS_SELECCIONADAS = [
    'DIA',

    # --- Consolidado KPI ---
    "EE Planta / Hl",
    "EE Elaboracion / Hl",
    "EE Bodega / Hl",
    "EE Cocina / Hl",
    #"EE Envasado / Hl",
    #"EE Servicios / Hl",
    #"EE Frio / Hl",
    #"EE Aire / Hl",
    #"EE CO2 / Hl",
    "EE Agua / Hl",
    #"Agua Planta / Hl",
    #"Agua Elab / Hl",
    #"Agua Bodega / Hl",
    #"Agua Cocina / Hl",
    #"Agua Envas / Hl",
    #"Agua Planta de Agua/Hl",
    #"Produccion Agua / Hl",
    "ET Planta / Hl",
    "ET Elab/Hl",
    "ET Bodega/Hl",
    "ET Cocina/Hl",
    "ET Envasado/Hl",
    #"Aire Planta / Hl",
    #"Aire Elaboracion / Hl",
    #"Aire Cocina / Hl",
    #"Aire Bodega / Hl",
    #"Aire Envasado / Hl",
    #"CO 2 / Hl",

    # --- Consolidado Produccion ---
    "Hl de Mosto",
    "Cocimientos Diarios",

    # --- Consolidado EE ---
    "Planta (Kw)",
    #"Elaboracion (Kw)",
    "Bodega (Kw)",
    #"Cocina (Kw)",
    #"Envasado (Kw)",
    #"Servicios (Kw)",
    #"Aire (Kw)",
    "Calderas (Kw)",
    "Efluentes (Kw)",
    "Frio (Kw)",  # Variable respuesta
    "Prod Agua (Kw)",

    # --- Totalizadores Energia ---
    "KW CO2",
    #"KW Enfluentes Coc",
    "KW Enfluente Efl",
    "KW Enfluentes Hidr",
    "Kw Compresores Aire",

    # --- Consolidado Agua (en hectolitros) ---
    #"Agua Planta (Hl)",
    #"Agua Elaboracion (Hl)",
    #"Agua Bodega (Hl)",
    #"Agua Cocina (Hl)",
    #"Agua Dilucion (Hl)",
    #"Agua Envasado (Hl)",
    #"Agua Servicios (Hl)",
    #"Planta de agua (Hl)",
    "Produccion (Hl)",
    #"Agua CO2",
    #"Agua Efluentes",

    # --- Totalizadores Agua ---
 #   "Agua Planta CO2",
    "Temp Tq Intermedio",

    # --- Consolidado GasVapor ---
#    "Conversion Kg/Mj",
    "Gas Planta (Mj)",
#    "Vapor Elaboracion (Kg)",
#  "Vapor Envasado (Kg)",
#    "Vapor Servicio (Kg)",
    "ET Envasado (Mj)",
    "ET Servicios (Mj)",
#    "Medicion Gas Planta (M3)",

    # --- Consolidado Aire ---
    #"Aire Producido (M3)",
    #"Aire Planta (M3)",
    #"Aire Elaboracion (m3)",
    #"Aire Envasado (M3)",
    #"Aire Servicios (M3)",

    # --- Totalizadores Glicol ---
    "Tot L3. L4 y Planta de CO2",
    "Tot A40/240/50/60/Centec/Filtro",
    "Tot  A130/330/430",
    "Tot  Trasiego"
]


COLUMNAS_PROMEDIO = [
]

COLUMNAS_ULTIMO = [

    # --- Consolidado KPI ---
    "EE Planta / Hl",
    "EE Elaboracion / Hl",
    "EE Bodega / Hl",
    "EE Cocina / Hl",
    #"EE Envasado / Hl",
    #"EE Servicios / Hl",
    #"EE Frio / Hl",
    #"EE Aire / Hl",
    #"EE CO2 / Hl",
    "EE Agua / Hl",
    #"Agua Planta / Hl",
    #"Agua Elab / Hl",
    #"Agua Bodega / Hl",
    #"Agua Cocina / Hl",
    #"Agua Envas / Hl",
    #"Agua Planta de Agua/Hl",
    #"Produccion Agua / Hl",
    "ET Planta / Hl",
    "ET Elab/Hl",
    "ET Bodega/Hl",
    "ET Cocina/Hl",
    "ET Envasado/Hl",
    #"Aire Planta / Hl",
    #"Aire Elaboracion / Hl",
    #"Aire Cocina / Hl",
    #"Aire Bodega / Hl",
    #"Aire Envasado / Hl",
    #"CO 2 / Hl",

    # --- Consolidado Produccion ---
    "Hl de Mosto",
    "Cocimientos Diarios",

    # --- Consolidado EE ---
    "Planta (Kw)",
    #"Elaboracion (Kw)",
    "Bodega (Kw)",
    #"Cocina (Kw)",
    #"Envasado (Kw)",
    #"Servicios (Kw)",
    #"Aire (Kw)",
    "Calderas (Kw)",
    "Efluentes (Kw)",
    "Frio (Kw)",  # Variable respuesta
    "Prod Agua (Kw)",

    # --- Totalizadores Energia ---
    "KW CO2",
    #"KW Enfluentes Coc",
    "KW Enfluente Efl",
    "KW Enfluentes Hidr",
    "Kw Compresores Aire",

    # --- Consolidado Agua (en hectolitros) ---
    #"Agua Planta (Hl)",
    #"Agua Elaboracion (Hl)",
    #"Agua Bodega (Hl)",
    #"Agua Cocina (Hl)",
    #"Agua Dilucion (Hl)",
    #"Agua Envasado (Hl)",
    #"Agua Servicios (Hl)",
    #"Planta de agua (Hl)",
    "Produccion (Hl)",
    #"Agua CO2",
    #"Agua Efluentes",

    # --- Totalizadores Agua ---
 #   "Agua Planta CO2",
    "Temp Tq Intermedio",

    # --- Consolidado GasVapor ---
#    "Conversion Kg/Mj",
    "Gas Planta (Mj)",
#    "Vapor Elaboracion (Kg)",
#  "Vapor Envasado (Kg)",
#    "Vapor Servicio (Kg)",
    "ET Envasado (Mj)",
    "ET Servicios (Mj)",
#    "Medicion Gas Planta (M3)",

    # --- Consolidado Aire ---
    #"Aire Producido (M3)",
    #"Aire Planta (M3)",
    #"Aire Elaboracion (m3)",
    #"Aire Envasado (M3)",
    #"Aire Servicios (M3)",

    # --- Totalizadores Glicol ---
    "Tot L3. L4 y Planta de CO2",
    "Tot A40/240/50/60/Centec/Filtro",
    "Tot  A130/330/430",
    "Tot  Trasiego"
]
