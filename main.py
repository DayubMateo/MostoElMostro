import subprocess
import sys
import os

print(">>> INICIANDO PIPELINE DE MACHINE LEARNING <<<")

scripts = [
    "Extractor_csv.py",
    "preprocess.ipynb",
    "EDA.ipynb"
]

for script in scripts:
    try:
        print(f"--> Ejecutando {script}...")

        if script.endswith(".py"):
            subprocess.run([sys.executable, script], check=True)

        elif script.endswith(".ipynb"):
            subprocess.run([
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute", script,
                "--output", script  # Sobrescribe el mismo archivo con la salida
            ], check=True)

        else:
            print(f"⚠️ Tipo de archivo no reconocido: {script}")

    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR al ejecutar {script}: {e}")
        break
    except FileNotFoundError:
        print(f"❌ ERROR: No se encontró el script {script}.")
        break

print(">>> PIPELINE FINALIZADO <<<")
