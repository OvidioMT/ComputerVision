const zonaCarga = document.getElementById('zonaCarga');
const inputImagen = document.getElementById('inputImagen');
const previsualizacion = document.getElementById('previsualizacion');
const textoZona = document.getElementById('textoZona');
const botonAnalizar = document.getElementById('botonAnalizar');
const selectModelo = document.getElementById('modelo');
const estadoDiv = document.getElementById('estado');
const resultadoDiv = document.getElementById('resultado');
const claseDetectada = document.getElementById('claseDetectada');
const confianzaTexto = document.getElementById('confianza');
const barrasDiv = document.getElementById('barras');
const segundaOpcion = document.getElementById('segundaOpcion');
const tiempoInferencia = document.getElementById('tiempoInferencia');
const accuracyModelo = document.getElementById('accuracyModelo');
const f1Modelo = document.getElementById('f1Modelo');
const seccionGradcam = document.getElementById('seccionGradcam');
const imagenOriginal = document.getElementById('imagenOriginal');
const imagenGradcam = document.getElementById('imagenGradcam');

let archivoSeleccionado = null;

function translateBackendError(message) {
  const errorMap = {
    'No se envio ninguna imagen.': 'No image was sent.',
    'No se envió ninguna imagen.': 'No image was sent.',
    'El archivo no es una imagen valida.': 'The uploaded file is not a valid image.',
    'El archivo no es una imagen válida.': 'The uploaded file is not a valid image.',
  };

  return errorMap[message] || message || 'Error processing image.';
}

zonaCarga.addEventListener('click', () => inputImagen.click());

inputImagen.addEventListener('change', (evento) => {
  const archivo = evento.target.files[0];
  if (archivo) manejarArchivo(archivo);
});

zonaCarga.addEventListener('dragover', (evento) => {
  evento.preventDefault();
  zonaCarga.classList.add('arrastrando');
});

zonaCarga.addEventListener('dragleave', () => {
  zonaCarga.classList.remove('arrastrando');
});

zonaCarga.addEventListener('drop', (evento) => {
  evento.preventDefault();
  zonaCarga.classList.remove('arrastrando');
  const archivo = evento.dataTransfer.files[0];
  if (archivo) manejarArchivo(archivo);
});

function manejarArchivo(archivo) {
  if (!archivo.type.startsWith('image/')) {
    mostrarEstadoImagen('Please select a valid image file.');
    return;
  }

  archivoSeleccionado = archivo;
  const lector = new FileReader();
  lector.onload = (evento) => {
    previsualizacion.src = evento.target.result;
    previsualizacion.hidden = false;
    textoZona.textContent = archivo.name;
  };
  lector.readAsDataURL(archivo);

  botonAnalizar.disabled = false;
  resultadoDiv.hidden = true;
  ocultarEstado();
}

botonAnalizar.addEventListener('click', async () => {
  if (!archivoSeleccionado) return;

  botonAnalizar.disabled = true;
  mostrarEstadoImagen('Analyzing image...');
  resultadoDiv.hidden = true;

  const datosFormulario = new FormData();
  datosFormulario.append('image', archivoSeleccionado);
  datosFormulario.append('modelo', selectModelo.value);

  try {
    const respuesta = await fetch('/api/predict', {
      method: 'POST',
      body: datosFormulario,
    });

    if (!respuesta.ok) {
      const datosError = await respuesta.json();
      throw new Error(translateBackendError(datosError.error));
    }

    const datos = await respuesta.json();
    mostrarResultado(datos);
    ocultarEstadoImagen();
  } catch (error) {
    mostrarEstadoImagen(error.message);
  } finally {
    botonAnalizar.disabled = false;
  }
});

function mostrarResultado(datos) {
  claseDetectada.textContent = datos.clase;
  confianzaTexto.textContent =
    `Confidence: ${(datos.confianza * 100).toFixed(1)}% - Model: ${datos.nombre_modelo}`;

  const margenPorcentaje = (datos.margen_top1_top2 * 100).toFixed(1);
  segundaOpcion.textContent =
    `Second option: ${datos.segunda_opcion.clase} ` +
    `(${(datos.segunda_opcion.confianza * 100).toFixed(1)}%) - ` +
    `difference of ${margenPorcentaje} points from the top prediction.`;

  tiempoInferencia.textContent = `${datos.tiempo_inferencia_ms} ms`;
  accuracyModelo.textContent = `${(datos.metricas_modelo.accuracy * 100).toFixed(2)}%`;
  f1Modelo.textContent = datos.metricas_modelo.f1_score.toFixed(4);

  if (datos.gradcam) {
    imagenOriginal.src = previsualizacion.src;
    imagenGradcam.src = datos.gradcam;
    seccionGradcam.hidden = false;
  } else {
    seccionGradcam.hidden = true;
  }

  barrasDiv.innerHTML = '';

  const entradas = Object.entries(datos.probabilidades)
    .sort((a, b) => b[1] - a[1]);

  for (const [nombreClase, probabilidad] of entradas) {
    const porcentaje = (probabilidad * 100).toFixed(1);
    const fila = document.createElement('div');
    fila.className = 'barra-fila';

    const esLaPredicha = nombreClase === datos.clase;

    fila.innerHTML = `
      <span class="barra-nombre">${nombreClase}</span>
      <span class="barra-pista">
        <span class="barra-relleno ${esLaPredicha ? 'activa' : ''}" style="width: ${porcentaje}%"></span>
      </span>
      <span class="barra-valor">${porcentaje}%</span>
    `;
    barrasDiv.appendChild(fila);
  }

  resultadoDiv.hidden = false;
}

function mostrarEstado(mensaje) {
  estadoDiv.textContent = mensaje;
  estadoDiv.hidden = false;
}

function mostrarEstadoImagen(mensaje) {
  estadoDiv.textContent = mensaje;
  estadoDiv.hidden = false;
}

function ocultarEstadoImagen() {
  estadoDiv.hidden = true;
}
