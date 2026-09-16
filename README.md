# **# Automatic License Plate Recognition (ALPR) Pipeline**

# 

Este proyecto implementa un pipeline automatizado de Inteligencia Artificial para la detección y lectura de placas vehiculares de forma local en CPU. Combina aprendizaje profundo para la localización geométrica y procesamiento digital de imágenes avanzado para optimizar el reconocimiento de texto (OCR).



El desarrollo de este sistema simula los flujos de un entorno laboral real, priorizando la curación rigurosa del dato y la experimentación morfológica sobre la complejidad inicial de los modelos.



\---



## \## 🏗️ Arquitectura de la Solución (End-to-End)



El sistema procesa las imágenes a través de tres fases desacopladas y secuenciales administradas por un único orquestador central (`main.py`):



1\. \*\*Localización de Objetos (Object Detection):\*\* Implementación de una arquitectura \*\*YOLOv8 Small\*\* entrenada localmente en CPU para predecir las coordenadas `xyxy` de la placa.

2\. \*\*Procesamiento Digital de Imágenes:\*\* Transformación matricial adaptativa que incluye escala de grises, escalado cúbico (`INTER\_CUBIC`).

3\. \*\*Reconocimiento Óptico de Caracteres (OCR):\*\* Extracción de texto mediante \*\*EasyOCR\*\* (PyTorch) con una capa final de lógica adaptativa y expresiones regulares en Python que impiden el \*hardcoding\* de formatos, se utiliza para la lectura OCR la imagen de color y en caso de lecturas erróneas una imagen transformada a escala de grises como respaldo.



\---



## \## 🔬 Laboratorio de Experimentación y Curación del Dato



Siguiendo principios de Ingeniería de Datos y validación empírica, el proyecto destaca por:

\* \*\*Evolución del Dataset:\*\* Identificación analítica de fallas tempranas de bajo umbral de confianza (1.27%), solucionadas mediante la expansión estratégica del dataset de 70 a \*\*321 imágenes\*\* y subiendo a \*\*30 épocas\*\*, logrando estabilizar la confianza por encima del \*\*80%\*\* en producción.

\* \*\*Estrategia de Fallback Dinámico:\*\* Gestión avanzada de casos fuera de distribución (Out-of-Distribution). El pipeline evalúa la calidad de la imagen a color y en caso de falla activa automáticamente un canal secundario sobre la matriz a escala de grises asegurar la lectura.

\* \*\*Bitácora en Notebooks:\*\* Registro histórico en la carpeta `notebooks/` que documenta la experimentación manual paso a paso con los filtros de OpenCV antes de su automatización final, reflejando el proceso analítico del ciclo de desarrollo.



\---



## \## 📂 Estructura del Repositorio



```text

license-plate-detector/

│

├── data/

│   ├── raw/                 # Imágenes originales sin procesar (Excluido en .gitignore)

│   ├── processed/           # Imágenes estandarizadas mediante script local (Excluido)

│   └── dataset/             # Splits de entrenamiento (70/20/10) y data.yaml (Excluido)

│

├── notebooks/

│   ├── .gitkeep             # Conserva la estructura de la carpeta para fases de prueba

│   └── experimentacion\_ocr.ipynb # Bitácora futura de experimentos morfológicos manuales (no generado en este momento)

│

├── runs/detect/             # Pesos y artefactos del entrenamiento local (Excluido en .gitignore)

├── scripts/

│   ├── data\_preprocessing.py # Script profesional de limpieza, validación y padding

│   ├── train\_detector.py     # Pipeline de entrenamiento optimizado para CPU (30 Epochs)

│   └── crop\_plates.py        # Inferencia local y recorte matricial NumPy

│

├── main.py                  # Orquestador definitivo del pipeline de punta a punta

├── requirements.txt         # Gestión optimizada de dependencias del entorno virtual

├── Protocolo de la Guía de Etiquetado de placas.txt		# Definición de la forma de etiquetar las placas

└── README.md                # Documentación técnica del portafolio

```



\---



## \## 📈 Resultados en Consola (Inferencia de Producción)



Al alimentar el orquestador unificado (`main.py`) con un activo vehicular, el sistema ejecuta y despliega la solución de forma automatizada:



```text

🤖 \[1/3] Localizando placa del vehículo con YOLO...

Ultralytics YOLOv8.4.150  Python-3.11.0 CPU (Intel Core...)

image 1/1 test\_car.jpg: 640x640 1 license-plate, 36.5ms

🛠️ \[2/3] Aplicando filtros morfológicos de la ruta modular...

🔤 \[3/3] Extrayendo caracteres con EasyOCR...



========================================

🎯 RECONOCIMIENTO AUTOMATIZADO DE PUNTA A PUNTA

========================================

📌 TEXTO DETECTADO:  DIS5BGT

📌 TEXTOS EXTRAS:    GUATEMALA 2004, CENTRO AMERICA

========================================

```



\---



## \## 🔄 Actualizaciones y mejoras



\* Durante las pruebas se observó fallas con tipografías delgadas o inclinadas, requiere ajuste en la lectura OCR

\* A pesar de tener umbral de confianza superior al 80%, el dataset utilizado era muy diverso en tipos de placas ya que contenía información de muchos países, es probable que se requiera un dataset especializado para un tipo de placa.



\---



\*Proyecto desarrollado con un enfoque analítico, reproducible y de código limpio.\*



