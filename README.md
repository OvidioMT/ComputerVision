# Brain Tumor Classifier

## Descripción

**Brain Tumor Classifier** es un proyecto de visión por computadora y aprendizaje profundo para clasificar tumores cerebrales a partir de imágenes de resonancia magnética (MRI). La aplicación utiliza redes neuronales preentrenadas mediante transfer learning para identificar cuatro clases principales:

- `glioma`
- `meningioma`
- `notumor`
- `pituitary`

El proyecto combina dos partes principales:

1. Un notebook de investigación y entrenamiento en [Proyecto2_IA.ipynb](Proyecto2_IA.ipynb)
2. Una aplicación web funcional para realizar predicciones desde una interfaz gráfica

---

## Características

* Clasificación automática de imágenes de MRI
* Soporte para dos modelos de transferencia:
  - ResNet50V2
  - EfficientNetB0
* Predicción con probabilidad por clase
* Visualización de la diferencia entre la primera y segunda clase predicha
* Mapa de activación Grad-CAM para interpretar la predicción
* Interfaz web simple y amigable
* Backend en Flask para servir la API y la aplicación

---

## Arquitectura del proyecto

```text
App-IA/
│
├── Proyecto2_IA.ipynb          # Notebook con entrenamiento, evaluación y análisis
├── READMEEJEMPLO.md            # Ejemplo de referencia para la documentación
├── README.md                   # Documentación principal del proyecto
│
├── backend/
│   ├── app.py                  # API Flask y lógica de inferencia
│   ├── requirements.txt        # Dependencias del backend
│   ├── models/                 # Modelos .keras entrenados
│   │   ├── resnet50v2_brain_tumor.keras
│   │   └── efficientnetb0_brain_tumor.keras
│   └── __pycache__/
│
└── frontend/
    ├── index.html              # Interfaz de usuario
    └── static/
        ├── css/
        │   └── style.css
        └── js/
            └── main.js
```

---

## Stack tecnológico

### Lenguaje

* Python

### Bibliotecas principales

* TensorFlow / Keras
* Flask
* NumPy
* OpenCV
* Pillow
* JavaScript
* HTML / CSS

### Modelos usados

* ResNet50V2
* EfficientNetB0

---

## Objetivos del proyecto

1. Cargar y procesar datasets de imágenes médicas MRI
2. Entrenar modelos con transfer learning
3. Comparar métricas entre arquitecturas
4. Evaluar desempeño con precisión, recall, F1-score y accuracy
5. Generar una app web capaz de clasificar imágenes en tiempo real
6. Visualizar regiones relevantes mediante Grad-CAM

---

## Requisitos previos

Antes de ejecutar el proyecto asegúrate de tener instalado:

* Python 3.10+
* pip
* entorno virtual (recomendado)
* acceso a la carpeta con los modelos entrenados

---

## Configuración del entorno

### 1. Clonar el repositorio

```bash
git clone https://github.com/nsandovalm/App-IA.git
cd App-IA
```

### 2. Crear un entorno virtual

En Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

En macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias del backend

```bash
cd backend
pip install -r requirements.txt
```

---

## Ejecución

### Iniciar el servidor backend

```bash
cd backend
python app.py
```

El proyecto quedará disponible en:

```text
http://localhost:5000
```

Luego abre el navegador y accede a esa dirección para usar la interfaz web.

---

## Uso de la aplicación

1. Selecciona el modelo a utilizar:
   - ResNet50V2
   - EfficientNetB0
2. Carga una imagen MRI en la interfaz
3. Haz clic en "Analyze image"
4. La app mostrará:
   - clase detectada
   - confianza de la predicción
   - distribución de probabilidades
   - tiempo de inferencia
   - comparación de resultados
   - mapa Grad-CAM (si está habilitado)

---

## Notas sobre los modelos

Los archivos de modelos entrenados se encuentran en la carpeta:

```text
backend/models/
```

Estos archivos tienen formato `.keras` y se cargan automáticamente al iniciar la aplicación. Si el proyecto se clona en otra máquina, es necesario que esos modelos estén presentes en la ruta indicada.

---

## Dataset

La investigación del proyecto se basa en un conjunto de imágenes MRI para la clasificación de tumores cerebrales. El notebook [Proyecto2_IA.ipynb](Proyecto2_IA.ipynb) describe la carga, exploración, balanceo y preparación del dataset.

El dataset original fue tomado de una fuente pública de Kaggle y contiene las siguientes cuatro categorías:

- `glioma`
- `meningioma`
- `notumor`
- `pituitary`

---

## Estructura funcional

### Notebook de investigación

El archivo [Proyecto2_IA.ipynb](Proyecto2_IA.ipynb) contiene:

* preparación del dataset
* exploración de clases
* análisis de distribución
* visualización de imágenes
* aumento de datos
* entrenamiento con ResNet50V2 y EfficientNetB0
* cálculo de métricas
* matrices de confusión
* curvas de aprendizaje
* Grad-CAM

### Backend

El archivo [backend/app.py](backend/app.py) contiene:

* carga de modelos
* preparación de la imagen de entrada
* predicción con Flask
* serialización de resultados JSON
* generación de Grad-CAM
* endpoint `/api/predict`

### Frontend

La interfaz en [frontend/index.html](frontend/index.html) y [frontend/static/js/main.js](frontend/static/js/main.js) permite:

* seleccionar el modelo
* subir una imagen
* enviar la imagen al backend
* visualizar resultados y métricas

---

## Recomendaciones futuras

* Añadir validación y pruebas automatizadas
* Mejorar el manejo de errores del backend
* Documentar la versión exacta de TensorFlow y dependencias
* Crear un archivo `.env` para configuración
* Preparar despliegue con Docker o un servidor WSGI
* Añadir soporte para múltiples tipos de imagen o preprocesamiento extra

---

## Créditos

### Autor

* Ovidio Martínez Taleno
