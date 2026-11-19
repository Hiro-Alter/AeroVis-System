# Evaluación de las capturas experimentales de la Prueba 2

La [ejecución conservada](../logs/prueba_2_capturas_repeticion_01/) contiene los CSV, el resumen de resultados y `run_info.json`, que documenta el entorno, el protocolo y las huellas de los archivos. Las imágenes y sus anotaciones se encuentran en el Anexo D del trabajo de grado.

Desde la raíz del repositorio, con las dependencias del aplicativo instaladas, indica la carpeta que contiene `annotations.json` y elige una carpeta nueva para la salida:

```powershell
python -m tests_future.evaluate_captures --dataset "ruta/al/dataset" --output "logs/prueba_2_nueva"
```

El evaluador procesa las 280 fotografías con ambos modelos y reutiliza los módulos de detección e inferencia del aplicativo. Los tiempos corresponden al reconocimiento a partir de una imagen en memoria, hasta obtener la etiqueta y su confianza.

