import subprocess
import sys

print(">>> INICIANDO PIPELINE DE MACHINE LEARNING <<<")

# Lista de los scripts a ejecutar en orden
scripts = [
    "Extractor_csv.py"
]

# Bucle para ejecutar cada script
for script in scripts:
    try:
        print(f"--> Ejecutando {script}...")
        # Llama a cada script y espera a que termine.
        # check=True asegura que si un script falla, el proceso se detiene.
        subprocess.run([sys.executable, script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR al ejecutar {script}: {e}")
        # Detiene la ejecución del pipeline si un paso falla
        break
    except FileNotFoundError:
        print(f"❌ ERROR: No se encontró el script {script}.")
        break

print(">>> PIPELINE FINALIZADO <<<")