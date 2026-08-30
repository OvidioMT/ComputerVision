# Brain Tumor Classifier

## Description

Brain Tumor Classifier is a computer vision and deep learning project for classifying brain tumors from magnetic resonance imaging (MRI) scans. The application uses pretrained neural networks via transfer learning to identify four main classes:

- `glioma`
- `meningioma`
- `notumor`
- `pituitary`

The project combines two main parts:

1. A research and training notebook in [Proyecto2_IA.ipynb](Proyecto2_IA.ipynb)
2. A functional web application for making predictions through a graphical interface

---

## Features

* Automatic classification of MRI images
* Support for two transfer learning models:
  - ResNet50V2
  - EfficientNetB0
* Prediction with per-class probability
* Visualization of the difference between the first and second predicted class
* Grad-CAM activation map to interpret the prediction
* Simple, user-friendly web interface
* Flask backend to serve the API and the application

---

## Project architecture

```text
ComputerVision/
│
├── Proyecto2_IA.ipynb          # Notebook with training, evaluation, and analysis
├── README.md                   # Main project documentation
│
├── backend/
│   ├── app.py                  # Flask API and inference logic
│   ├── requirements.txt        # Backend dependencies
│   └── models/                 # Trained .keras models
│       ├── resnet50v2_brain_tumor.keras
│       └── efficientnetb0_brain_tumor.keras
│   
│
└── frontend/
    ├── index.html              # User interface
    └── static/
        ├── css/
        │   └── style.css
        └── js/
            └── main.js
```

---

## Tech stack

### Language

* Python

### Main libraries

* TensorFlow / Keras
* Flask
* NumPy
* OpenCV
* Pillow
* JavaScript
* HTML / CSS

### Models used

* ResNet50V2
* EfficientNetB0

---

## Project goals

1. Load and process medical MRI image datasets
2. Train models using transfer learning
3. Compare metrics between architectures
4. Evaluate performance using precision, recall, F1-score, and accuracy
5. Build a web app capable of classifying images in real time
6. Visualize relevant regions using Grad-CAM

---

## Prerequisites

Before running the project, make sure you have installed:

* Python 3.10+
* pip
* virtual environment (recommended)
* access to the folder with the trained models

---

## Environment setup

### 1. Clone the repository

```bash
git clone https://github.com/OvidioMT/ComputerVision.git
cd ComputerVision
```

### 2. Create a virtual environment with Python 3.12

On Windows, Python 3.12:

```bash
py -3.12 -m venv venv
venv\Scripts\activate
```

On macOS / Linux:

```bash
python3.12 -m venv venv
source venv/bin/activate
```

If python3.12 is not available on your system, install Python 3.12 before continuing.

### 3. Install backend dependencies

```bash
cd backend
pip install -r requirements.txt
```

---

## Running the project

### Start the backend server

```bash
cd backend
python app.py
```

The project will be available at:

```text
http://localhost:5000
```

---

## Using the application

1. Select the model to use:
   - ResNet50V2
   - EfficientNetB0
2. Upload an MRI image in the interface
3. Click "Analyze image"
4. The app will display:
   - detected class
   - prediction confidence
   - probability distribution
   - inference time
   - results comparison
   - Grad-CAM map (if enabled)

---

## Notes on the models

The trained model files are located in the folder:

```text
backend/models/
```

These files are in `.keras` format and are loaded automatically when the application starts. If the project is cloned on another machine, those models must be present at the indicated path.

---

## Dataset

The project's research is based on a set of MRI images for brain tumor classification. The notebook [Proyecto2_IA.ipynb](Proyecto2_IA.ipynb) describes the loading, exploration, balancing, and preparation of the dataset.

The original dataset was taken from a public Kaggle source and contains the following four categories:

- `glioma`
- `meningioma`
- `notumor`
- `pituitary`

---

## Functional structure

### Research notebook

The file [Proyecto2_IA.ipynb](Proyecto2_IA.ipynb) contains:

* dataset preparation
* class exploration
* distribution analysis
* image visualization
* data augmentation
* training with ResNet50V2 and EfficientNetB0
* metrics calculation
* confusion matrices
* learning curves
* Grad-CAM

### Backend

The file [backend/app.py](backend/app.py) contains:

* model loading
* input image preparation
* prediction with Flask
* JSON result serialization
* Grad-CAM generation
* `/api/predict` endpoint

### Frontend

The interface in [frontend/index.html](frontend/index.html) and [frontend/static/js/main.js](frontend/static/js/main.js) allows you to:

* select the model
* upload an image
* send the image to the backend
* view results and metrics

---

## Credits

### Author

* Ovidio Martínez Taleno
