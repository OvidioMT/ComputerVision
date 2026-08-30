
import os
import time
import base64
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from tensorflow import keras
from PIL import Image
import io

#Configuración
IMG_SIZE = 224
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), 'frontend')

RESNET_PATH = os.path.join(MODELS_DIR, 'resnet50v2_brain_tumor.keras')
EFFNET_PATH = os.path.join(MODELS_DIR, 'efficientnetb0_brain_tumor.keras')

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')

# Carga de modelos
print('Cargando modelos, esto puede tardar unos segundos...')
modelo_resnet = keras.models.load_model(RESNET_PATH)
modelo_eff = keras.models.load_model(EFFNET_PATH)
print('Modelos cargados correctamente.')

MODELOS = {
    'resnet': modelo_resnet,
    'efficientnet': modelo_eff,
}


METRICAS_MODELOS = {
    'resnet': {
        'nombre_completo': 'ResNet50V2',
        'accuracy': 0.9238,
        'precision': 0.9267,
        'recall': 0.9238,
        'f1_score': 0.9224,
    },
    'efficientnet': {
        'nombre_completo': 'EfficientNetB0',
        'accuracy': 0.8725,
        'precision': 0.8686,
        'recall': 0.8725,
        'f1_score': 0.8686,
    },
}


def preparar_imagen(imagen_pil: Image.Image, nombre_modelo: str) -> np.ndarray:
    imagen_pil = imagen_pil.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    img_array = keras.utils.img_to_array(imagen_pil)

    if nombre_modelo == 'resnet':
        img_array = img_array / 255.0

    return np.expand_dims(img_array, axis=0)


#Grad CAM
def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if hasattr(layer, 'layers'):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return sub_layer.name, layer
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name, None
    raise ValueError('No se encontró ninguna capa Conv2D en el modelo.')


def calcular_grad_cam(model, img_array, conv_layer_name, base_model):
    inputs = tf.cast(img_array, tf.float32)

    if base_model is not None:
        conv_output_tensor = base_model.get_layer(conv_layer_name).output
        grad_model = keras.Model(
            inputs=base_model.inputs,
            outputs=[conv_output_tensor, base_model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, base_predictions = grad_model(inputs, training=False)
            x = base_predictions
            base_layer_found = False
            for layer in model.layers:
                if layer == base_model:
                    base_layer_found = True
                    continue
                if base_layer_found:
                    x = layer(x)
            predictions = x
            pred_index = tf.argmax(predictions[0])
            class_score = predictions[:, pred_index]
    else:
        conv_output_tensor = model.get_layer(conv_layer_name).output
        grad_model = keras.Model(
            inputs=model.inputs,
            outputs=[conv_output_tensor, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(inputs, training=False)
            pred_index = tf.argmax(predictions[0])
            class_score = predictions[:, pred_index]

    grads = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), predictions.numpy()[0]


def generar_imagen_gradcam_base64(modelo, img_array_modelo, img_array_original_255):
    """
    Genera la imagen de superposición Grad-CAM y la devuelve como
    data URL base64 lista para insertar en un <img src="..."> del frontend.
    """
    conv_name, base_model = get_last_conv_layer_name(modelo)
    heatmap, _ = calcular_grad_cam(modelo, img_array_modelo, conv_name, base_model)

    img_display = img_array_original_255 / 255.0
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0

    superpuesta = np.clip(0.6 * img_display + 0.4 * heatmap_colored, 0, 1)
    superpuesta_uint8 = (superpuesta * 255).astype(np.uint8)

    imagen_resultado = Image.fromarray(superpuesta_uint8)
    buffer = io.BytesIO()
    imagen_resultado.save(buffer, format='PNG')
    codificada = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return f'data:image/png;base64,{codificada}'


@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No se envió ninguna imagen.'}), 400

    archivo = request.files['image']
    modelo_elegido = request.form.get('modelo', 'resnet')
    incluir_gradcam = request.form.get('gradcam', 'true').lower() == 'true'

    if modelo_elegido not in MODELOS:
        return jsonify({'error': f'Modelo "{modelo_elegido}" no reconocido.'}), 400

    try:
        imagen_pil = Image.open(io.BytesIO(archivo.read()))
    except Exception:
        return jsonify({'error': 'El archivo no es una imagen válida.'}), 400

    modelo = MODELOS[modelo_elegido]
    entrada = preparar_imagen(imagen_pil, modelo_elegido)

    inicio = time.perf_counter()
    probabilidades = modelo.predict(entrada, verbose=0)[0]
    tiempo_inferencia_ms = (time.perf_counter() - inicio) * 1000

    indices_ordenados = np.argsort(probabilidades)[::-1]
    indice_top1 = int(indices_ordenados[0])
    indice_top2 = int(indices_ordenados[1])
    margen = float(probabilidades[indice_top1] - probabilidades[indice_top2])

    nivel_certeza = 'alta' if margen > 0.4 else ('media' if margen > 0.15 else 'baja')

    resultado = {
        'clase': CLASS_NAMES[indice_top1],
        'confianza': float(probabilidades[indice_top1]),
        'modelo_usado': modelo_elegido,
        'nombre_modelo': METRICAS_MODELOS[modelo_elegido]['nombre_completo'],
        'probabilidades': {
            CLASS_NAMES[i]: float(probabilidades[i]) for i in range(len(CLASS_NAMES))
        },
        'segunda_opcion': {
            'clase': CLASS_NAMES[indice_top2],
            'confianza': float(probabilidades[indice_top2]),
        },
        'margen_top1_top2': margen,
        'nivel_certeza': nivel_certeza,
        'tiempo_inferencia_ms': round(tiempo_inferencia_ms, 1),
        'metricas_modelo': METRICAS_MODELOS[modelo_elegido],
    }

    if incluir_gradcam:
        try:
            img_para_visual = keras.utils.img_to_array(
                imagen_pil.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
            )
            resultado['gradcam'] = generar_imagen_gradcam_base64(
                modelo, entrada, img_para_visual
            )
        except Exception as error:
            resultado['gradcam'] = None
            resultado['gradcam_error'] = str(error)

    return jsonify(resultado)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
