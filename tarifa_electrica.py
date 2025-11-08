import pandas as pd
import datetime

# 1. Definir periodos
periodos = [
    {
        "inicio": (2020, 5, 1), "fin": (2020, 10, 25),  # del 1 de mayo al 25 de octubre (sábado antes del último domingo)
        "horas": {
            "lunes":   {"base": 0, "intermedia": 20, "punta": 4},
            "sabado":  {"base": 0, "intermedia": 24, "punta": 0},
            "domingo": {"base": 0, "intermedia": 24, "punta": 0},
        }
    },
    {
        "inicio": (2020, 10, 25), "fin": (2021, 4, 30), # del último domingo de octubre al 30 de abril
        "horas": {
            "lunes":   {"base": 19, "intermedia": 5, "punta": 0},
            "sabado":  {"base": 21, "intermedia": 3, "punta": 0},
            "domingo": {"base": 24, "intermedia": 0, "punta": 0},
        }
    },
    {
        "inicio": (2021, 4, 30), "fin": (2021, 10, 30),
        "horas": {
            "lunes":   {"base": 0, "intermedia": 20, "punta": 4},
            "sabado":  {"base": 0, "intermedia": 24, "punta": 0},
            "domingo": {"base": 0, "intermedia": 24, "punta": 0},
        }
    },
    {
        "inicio": (2021, 10, 30), "fin": (2022, 4, 30),
        "horas": {
            "lunes":   {"base": 19, "intermedia": 5, "punta": 0},
            "sabado":  {"base": 21, "intermedia": 3, "punta": 0},
            "domingo": {"base": 24, "intermedia": 0, "punta": 0},
        }
    },
    {
        "inicio": (2022, 4, 30), "fin": (2022, 10, 30),
        "horas": {
            "lunes":   {"base": 0, "intermedia": 20, "punta": 4},
            "sabado":  {"base": 0, "intermedia": 24, "punta": 0},
            "domingo": {"base": 0, "intermedia": 24, "punta": 0},
        }
    },
    {
        "inicio": (2022, 10, 30), "fin": (2023, 4, 30),
        "horas": {
            "lunes":   {"base": 19, "intermedia": 5, "punta": 0},
            "sabado":  {"base": 21, "intermedia": 3, "punta": 0},
            "domingo": {"base": 24, "intermedia": 0, "punta": 0},
        }
    },
    {
        "inicio": (2023, 4, 30), "fin": (2023, 10, 30),
        "horas": {
            "lunes":   {"base": 0, "intermedia": 20, "punta": 4},
            "sabado":  {"base": 0, "intermedia": 24, "punta": 0},
            "domingo": {"base": 0, "intermedia": 24, "punta": 0},
        }
    },
    {
        "inicio": (2023, 10, 30), "fin": (2024, 4, 30),
        "horas": {
            "lunes":   {"base": 19, "intermedia": 5, "punta": 0},
            "sabado":  {"base": 21, "intermedia": 3, "punta": 0},
            "domingo": {"base": 24, "intermedia": 0, "punta": 0},
        }
    }
]

# --- 2. Cargos por año y mes ---
cargos_por_anio = {
    2020: {
        1: {"fijo": 697.99, "base": 0.5492, "intermedia": 0.8473, "punta": 0.0000},
        2: {"fijo": 697.99, "base": 0.5464, "intermedia": 0.8423, "punta": 0.0000},
        3: {"fijo": 697.99, "base": 0.5488, "intermedia": 0.8466, "punta": 0.0000},
        4: {"fijo": 697.99, "base": 0.5512, "intermedia": 0.8508, "punta": 0.0000},
        5: {"fijo": 697.99, "base": 0.5471, "intermedia": 0.84355, "punta": 1.1347},
        6: {"fijo": 697.99, "base": 0.5437, "intermedia": 0.8372, "punta": 1.1256},
        7: {"fijo": 697.99, "base": 0.5350, "intermedia": 0.8215, "punta": 1.1030},
        8: {"fijo": 697.99, "base": 0.5297, "intermedia": 0.8119, "punta": 1.0891},
        9: {"fijo": 697.99, "base": 0.5197, "intermedia": 0.7939, "punta": 1.0633},
        10: {"fijo": 697.99, "base": 0.5198, "intermedia": 0.7940, "punta": 1.0634},
        11: {"fijo": 697.99, "base": 0.5199, "intermedia": 0.7942, "punta": 0.0000},
        12: {"fijo": 697.99, "base": 0.5195, "intermedia": 0.7934, "punta": 0.0000},
    },
    2021: {
        1: {"fijo": 718.93, "base": 0.5298, "intermedia": 0.8069, "punta": 0.0000},
        2: {"fijo": 718.93, "base": 0.5321, "intermedia": 0.8111, "punta": 0.0000},
        3: {"fijo": 718.93, "base": 0.5329, "intermedia": 0.8126, "punta": 0.0000},
        4: {"fijo": 718.93, "base": 0.5458, "intermedia": 0.8360, "punta": 0.0000},
        5: {"fijo": 718.93, "base": 0.5477, "intermedia": 0.8393, "punta": 1.1259},
        6: {"fijo": 718.93, "base": 0.5615, "intermedia": 0.8643, "punta": 1.1618},
        7: {"fijo": 718.93, "base": 0.5571, "intermedia": 0.8563, "punta": 1.1503},
        8: {"fijo": 718.93, "base": 0.5564, "intermedia": 0.8550, "punta": 1.1485},
        9: {"fijo": 718.93, "base": 0.5479, "intermedia": 0.8397, "punta": 1.1264},
        10: {"fijo": 718.93, "base": 0.5469, "intermedia": 0.8380, "punta": 1.1239},
        11: {"fijo": 718.93, "base": 0.5503, "intermedia": 0.8441, "punta": 1.1327},
        12: {"fijo": 718.93, "base": 0.5549, "intermedia": 0.8524, "punta": 1.1446},
    },
    2022: {
        1: {"fijo": 535.80, "base": 0.5603, "intermedia": 0.8611, "punta": 1.1566},
        2: {"fijo": 535.80, "base": 0.5635, "intermedia": 0.8669, "punta": 1.1649},
        3: {"fijo": 535.80, "base": 0.5635, "intermedia": 0.8669, "punta": 1.1649},
        4: {"fijo": 535.80, "base": 0.5747, "intermedia": 0.8872, "punta": 1.1942},
        5: {"fijo": 535.80, "base": 0.5834, "intermedia": 0.9030, "punta": 1.2169},
        6: {"fijo": 535.80, "base": 0.5711, "intermedia": 0.8807, "punta": 1.1848},
        7: {"fijo": 535.80, "base": 0.5822, "intermedia": 0.9007, "punta": 1.2136},
        8: {"fijo": 535.80, "base": 0.5844, "intermedia": 0.9047, "punta": 1.2194},
        9: {"fijo": 535.80, "base": 0.5866, "intermedia": 0.9087, "punta": 1.2252},
        10: {"fijo": 535.80, "base": 0.5882, "intermedia": 0.9116, "punta": 1.2294},
        11: {"fijo": 535.80, "base": 0.5892, "intermedia": 0.9135, "punta": 1.2321},
        12: {"fijo": 535.80, "base": 0.5892, "intermedia": 0.9135, "punta": 1.2321},
    },
    2023: {
        1: {"fijo": 598.55, "base": 0.5895, "intermedia": 0.9148, "punta": 1.2343},
        2: {"fijo": 598.55, "base": 0.5934, "intermedia": 0.9218, "punta": 1.2444},
        3: {"fijo": 598.55, "base": 0.6008, "intermedia": 0.9353, "punta": 1.2638},
        4: {"fijo": 598.55, "base": 0.6067, "intermedia": 0.9459, "punta": 1.2791},
        5: {"fijo": 598.55, "base": 0.6171, "intermedia": 0.9647, "punta": 1.3062},
        6: {"fijo": 598.55, "base": 0.6243, "intermedia": 0.9777, "punta": 1.3249},
        7: {"fijo": 598.55, "base": 0.6305, "intermedia": 0.9890, "punta": 1.3412},
        8: {"fijo": 598.55, "base": 0.6261, "intermedia": 0.9810, "punta": 1.3297},
        9: {"fijo": 598.55, "base": 0.6218, "intermedia": 0.9731, "punta": 1.3183},
        10: {"fijo": 598.55, "base": 0.6224, "intermedia": 0.9743, "punta": 1.3201},
        11: {"fijo": 598.55, "base": 0.6166, "intermedia": 0.9638, "punta": 1.3048},
        12: {"fijo": 598.55, "base": 0.6133, "intermedia": 0.9577, "punta": 1.2962},
    },
    2024: {
        1: {"fijo": 631.59, "base": 0.6168, "intermedia": 0.9631, "punta": 1.3033},
        2: {"fijo": 631.59, "base": 0.6193, "intermedia": 0.9676, "punta": 1.3098},
        3: {"fijo": 631.59, "base": 0.6247, "intermedia": 0.9774, "punta": 1.3239},
        4: {"fijo": 631.59, "base": 0.6247, "intermedia": 0.9774, "punta": 1.3239},
        5: {"fijo": 631.59, "base": 0.6247, "intermedia": 0.9774, "punta": 1.3239},
        6: {"fijo": 631.59, "base": 0.6247, "intermedia": 0.9774, "punta": 1.3239},
        7: {"fijo": 631.59, "base": 0.6370, "intermedia": 0.9997, "punta": 1.3560},
        8: {"fijo": 631.59, "base": 0.6409, "intermedia": 1.0067, "punta": 1.3661},
        9: {"fijo": 631.59, "base": 0.6409, "intermedia": 1.0068, "punta": 1.3662},
        10: {"fijo": 631.59, "base": 0.6409, "intermedia": 1.0068, "punta": 1.3662},
        11: {"fijo": 631.59, "base": 0.6409, "intermedia": 1.0068, "punta": 1.3662},
        12: {"fijo": 631.59, "base": 0.6334, "intermedia": 0.9932, "punta": 1.3466},
    },
    2025: {
        1: {"fijo": 782.90, "base": 0.6426, "intermedia": 1.0065, "punta": 1.3640},
        2: {"fijo": 782.90, "base": 0.6433, "intermedia": 1.0078, "punta": 1.3659},
        3: {"fijo": 782.90, "base": 0.6435, "intermedia": 1.0081, "punta": 1.3663},
        4: {"fijo": 782.90, "base": 0.6435, "intermedia": 1.0081, "punta": 1.3663},
        5: {"fijo": 782.90, "base": 0.6435, "intermedia": 1.0081, "punta": 1.3663},
        6: {"fijo": 782.90, "base": 0.6541, "intermedia": 1.0275, "punta": 1.3942},
        7: {"fijo": 782.90, "base": 0.6610, "intermedia": 1.0400, "punta": 1.4122},
        8: {"fijo": 782.90, "base": 0.6617, "intermedia": 1.0411, "punta": 1.4139},
        9: {"fijo": 782.90, "base": 0.6615, "intermedia": 1.0375, "punta": 1.4068},
        10: {"fijo": 782.90, "base": 0.6615, "intermedia": 1.0375, "punta": 1.4069},
        11: {"fijo": 782.90, "base": 0.6476, "intermedia": 1.0123, "punta": 1.3705},
        12: {"fijo": 782.90, "base": 0.6476, "intermedia": 1.0123, "punta": 1.3705},
    },
}


# 3. Columnas de consumo a sumar


# --- 4. Función general actualizada ---
# --- 4. Función general actualizada ---
def calcular_tarifa_electrica_general(df, periodos, cargos_por_anio):
    # Obtenemos las columnas de consumo en KW
    columnas_consumo = [col for col in df.columns if "KW" in col.upper()]
    
    """
    Calcula la tarifa eléctrica diaria con cargos variables por mes y año.
    """

    def obtener_periodo_actual(anio, mes, dia):
        fecha = datetime.date(anio, mes, dia)
        for p in periodos:
            inicio = datetime.date(*p["inicio"])
            fin = datetime.date(*p["fin"])
            if inicio <= fecha <= fin:
                return p
        return periodos[-1]

    def obtener_cargos(anio, mes):
        """Devuelve los cargos aplicables al año y mes. Si no hay, toma el último disponible."""
        if anio in cargos_por_anio:
            if mes in cargos_por_anio[anio]:
                return cargos_por_anio[anio][mes]
            else:
                return list(cargos_por_anio[anio].values())[-1]
        else:
            return list(cargos_por_anio.values())[-1][12]

    def calcular_tarifa_fila(row):
        anio, mes, dia = row["Anio"], row["Mes"], row["Dia"]
        dia_semana = row["Dia_semana"].lower()

        # Normalización del día
        if "sáb" in dia_semana or "sab" in dia_semana:
            dia_semana = "sabado"
        elif "dom" in dia_semana:
            dia_semana = "domingo"
        else:
            dia_semana = "lunes"

        # Obtener periodo y cargos
        periodo = obtener_periodo_actual(anio, mes, dia)
        cargos = obtener_cargos(anio, mes)
        horas = periodo["horas"][dia_semana]

        # Consumo diario total
        consumo_kw = row[columnas_consumo].sum()

        # Tarifa ponderada
        tarifa_variable = (
            (horas["base"] / 24) * cargos["base"] +
            (horas["intermedia"] / 24) * cargos["intermedia"] +
            (horas["punta"] / 24) * cargos["punta"]
        )

        return (cargos["fijo"] + tarifa_variable) * consumo_kw

    df["Tarifa_electrica"] = df.apply(calcular_tarifa_fila, axis=1)
    return df