MostoElMostro - Sistema de Predicción
Sistema de machine learning para análisis y predicción de datos.

🚀 Instalación Rápida
Prerrequisitos
Git

Python 3.8+

UV package manager

Pasos de instalación
Clonar el repositorio
```
git clone https://github.com/DayubMateo/MostoElMostro.git
cd MostoElMostro
```

Crear y activar entorno virtual
```
uv venv
```

Activar entorno virtual:

Windows:
```
.venv\Scripts\activate
```

Linux/Mac:
```
source .venv/bin/activate
```

Instalar dependencias
```
uv pip install -r requirements.txt
```

📊 Flujo de Trabajo
1. Preprocesamiento de Datos
```
python src/preprocessing_pipeline.py
```
⏰ Tiempo estimado: 4 minutos

2. (Opcional) Análisis Exploratorio
Ejecutar los notebooks (.ipynb) para análisis:

Seleccionar el kernel de Python del entorno virtual creado

Ejecutar celdas en orden

3. Entrenar el Modelo
```
python src/train_model.py
```

4. Realizar Predicciones
```
python src/predict.py "ruta/al/archivo.xlsx"
```

📁 Estructura de Archivos
```
MostoElMostro/
├── src/ # Código fuente
│ ├── preprocessing_pipeline.py
│ ├── train_model.py
│ └── predict.py
├── results/ # Resultados
│ └── predicciones.csv # Predicciones generadas
├── notebooks/ # Análisis exploratorio
└── requirements.txt # Dependencias
```

🎯 Uso
Preparar datos: Ejecutar el pipeline de preprocesamiento

Entrenar modelo: Guardar el mejor modelo con train_model.py

Predecir: Usar predict.py con la ruta de tu archivo Excel

Resultados: Encontrar las predicciones en results/predicciones.csv

💡 Notas Importantes
Asegúrate de tener el entorno virtual activado antes de ejecutar cualquier script

El preprocesamiento es necesario antes del entrenamiento del modelo

Los notebooks usan el kernel del entorno virtual creado

🆘 Troubleshooting
Problema: Error al activar entorno virtual
Solución: Ejecutar ```Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser``` en PowerShell (Windows)

Problema: No se encuentra el kernel en notebooks
Solución: Seleccionar manualmente el kernel de Python del entorno virtual (.venv)

¿Necesitas ayuda? Revisa los notebooks para ejemplos detallados de uso.

1. Dentro de la carpeta en la que se desea almacenar el proyecto, ejecutar el siguiente comando:"git clone https://github.com/DayubMateo/MostoElMostro.git"
2. Abrir la terminal con Ctrl + Ñ
3. Crear el entorno virtual con "uv venv"
4. Activar el entorno virtual copiando la ruta de '.venv/Scripts/activate', pegándola en la terminal y presionar Enter
5. Instalar dependencias con "uv pip install -r requirements.txt"	
6. Ejecutar el pipeline de preprocesamiento para los 3 archivos iniciales mediante 'python src/preprocessing_pipeline.py' (tarda aproximadamente unos 4 minutos)
7. (Opcional) Ejecutar los notebooks (terminados en .ipynb)
7.1 Puede surgir una notificación hablando sobre el kernel, seleccionar el kernel de Python con el nombre del entorno virtual
7.2 Puede surgir una notificación solicitando
8. Guardar el mejor modelo con "python src/train_model.py"
9. Ponerlo a prueba mediante 'python src/predict.py "<ruta_del_excel>"'
10. Los resultados serán almacenados en "results/predicciones.csv"