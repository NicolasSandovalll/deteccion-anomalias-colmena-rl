/**
 * app.js
 * ---------------------------------------------------------------------------
 * Orquestador del entrenamiento del agente de Aprendizaje por Refuerzo.
 *
 * Une el simulador de colmena (que genera episodios de lecturas) con el agente
 * Q-Learning: normaliza las lecturas, entrena al agente sobre muchos episodios,
 * lo evalúa en episodios nuevos y muestra las métricas por consola.
 * ---------------------------------------------------------------------------
 */

'use strict';

const {
  crearEpisodio,
  getHoraDelDia,
  PASOS_POR_EPISODIO,
  RANGOS_SENSORES
} = require('./simulador');

const {
  QAgent,
  discretizarEstado,
  calcularRecompensa,
  NUM_ESTADOS,
  NUM_ACCIONES,
  ACCIONES,
  ANOMALIAS_GRAVES,
  ANOMALIAS_LEVES
} = require('./qlearning');

// ===========================================================================
// 1) NORMALIZACIÓN DE SENSORES
// ===========================================================================

/**
 * Normaliza cada lectura cruda a [0, 1] usando el rango [min, max] de su sensor.
 *
 * @param {{temp:number, humedad:number, vibracion:number, flujo:number}} lecturas
 * @param {object} rangos - RANGOS_SENSORES: {sensor: {min, max}}
 * @returns {{tempNorm:number, humedadNorm:number, vibracionNorm:number, flujoNorm:number}}
 */
function normalizarSensores(lecturas, rangos) {
  // Escala un valor al [0,1] de su rango y lo recorta por si sale de los límites.
  const norm = (valor, sensor) => {
    const { min, max } = rangos[sensor];
    const v = (valor - min) / (max - min);
    return Math.min(1, Math.max(0, v));
  };

  return {
    tempNorm:      norm(lecturas.temp,      'temp'),
    humedadNorm:   norm(lecturas.humedad,   'humedad'),
    vibracionNorm: norm(lecturas.vibracion, 'vibracion'),
    flujoNorm:     norm(lecturas.flujo,     'flujo')
  };
}

// ===========================================================================
// UTILIDAD: derivar el índice de estado de un paso del episodio
// ===========================================================================

/**
 * Convierte una observación cruda del episodio en el índice de estado discreto
 * que entiende el agente.
 *
 * La temperatura y la humedad se discretizan en unidades físicas, mientras que
 * vibración y flujo se normalizan a [0, 1] antes de discretizar.
 *
 * @param {object} obs - elemento del episodio {paso, hora, lecturas, anomalia}
 * @param {boolean} alertaDisparada
 * @returns {number} índice de estado
 */
function estadoDeObservacion(obs, alertaDisparada) {
  const n = normalizarSensores(obs.lecturas, RANGOS_SENSORES);
  const hora = getHoraDelDia(obs.paso);
  return discretizarEstado(
    obs.lecturas.temp,     // temp en °C
    obs.lecturas.humedad,  // humedad en %
    n.vibracionNorm,       // vibración normalizada
    n.flujoNorm,           // flujo normalizado
    hora,
    alertaDisparada ? 1 : 0
  );
}

/**
 * Calcula el valor de alertaDisparada para el paso siguiente:
 *   - se activa si el agente ha emitido cualquier alerta (acción > observar),
 *   - se reinicia cuando el paso actual no tiene anomalía (la anomalía terminó).
 *
 * @param {boolean} alertaActual
 * @param {number} accion
 * @param {string|null} tipoAnomalia
 * @returns {boolean}
 */
function siguienteAlerta(alertaActual, accion, tipoAnomalia) {
  let alerta = alertaActual;
  if (accion > ACCIONES.OBSERVAR) alerta = true;
  if (tipoAnomalia == null) alerta = false;
  return alerta;
}

// ===========================================================================
// 2) ENTRENAMIENTO
// ===========================================================================

/**
 * Entrena al agente sobre numEpisodios episodios generados por el simulador.
 *
 * En cada paso: discretiza el estado, el agente elige acción (epsilon-greedy),
 * se calcula la recompensa real, se actualiza la tabla Q hacia el estado
 * siguiente y se propaga el flag alertaDisparada. Al terminar cada episodio se
 * aplica el decay de epsilon y se registra la recompensa acumulada.
 *
 * @param {QAgent} agente
 * @param {number} numEpisodios
 * @returns {{agente:QAgent, historialRecompensas:number[]}}
 */
function entrenar(agente, numEpisodios = 1000) {
  const historialRecompensas = [];

  for (let ep = 0; ep < numEpisodios; ep++) {
    const episodio = crearEpisodio();
    let alertaDisparada = false;
    let recompensaEpisodio = 0;

    for (let i = 0; i < episodio.length; i++) {
      const obs = episodio[i];

      // Estado actual y acción del agente.
      const estado = estadoDeObservacion(obs, alertaDisparada);
      const accion = agente.seleccionarAccion(estado);

      // Recompensa real de este paso (usa la alerta ANTES de actualizarla).
      const recompensa = calcularRecompensa(accion, obs.anomalia, alertaDisparada);
      recompensaEpisodio += recompensa;

      // Flag de alerta que regirá en el paso siguiente.
      const nuevaAlerta = siguienteAlerta(alertaDisparada, accion, obs.anomalia);

      // Estado siguiente (el último paso reutiliza el propio estado como cierre).
      const obsSiguiente = episodio[i + 1];
      const estadoSiguiente = obsSiguiente
        ? estadoDeObservacion(obsSiguiente, nuevaAlerta)
        : estado;

      // Aprendizaje.
      agente.actualizarQ(estado, accion, recompensa, estadoSiguiente);

      alertaDisparada = nuevaAlerta;
    }

    // Menos exploración a medida que avanza el entrenamiento.
    agente.decayEpsilon();
    historialRecompensas.push(recompensaEpisodio);
  }

  return { agente, historialRecompensas };
}

// ===========================================================================
// 3) EVALUACIÓN
// ===========================================================================

/**
 * Ejecuta un episodio en modo explotación pura (epsilon = 0, sin actualizar Q)
 * y contabiliza el desempeño del agente.
 *
 * Clasificación de cada paso (aciertos + falsoPositivos + falsoNegativos cubre
 * todos los pasos):
 *   - acierto        : observar sin anomalía, o alertar con la severidad correcta
 *                      (alarma ante grave, aviso ante leve).
 *   - falsoPositivo  : emitir cualquier alerta sin que haya anomalía.
 *   - falsoNegativo  : no atender bien una anomalía real (no alertar, o alertar
 *                      con la severidad equivocada).
 *
 * @param {QAgent} agente
 * @param {Array} episodio - salida de crearEpisodio()
 * @returns {{aciertos:number, falsoPositivos:number, falsoNegativos:number, recompensaTotal:number}}
 */
function evaluarEpisodio(agente, episodio) {
  // Forzar explotación sin perder el epsilon aprendido.
  const epsilonPrevio = agente.epsilon;
  agente.epsilon = 0;

  let aciertos = 0;
  let falsoPositivos = 0;
  let falsoNegativos = 0;
  let recompensaTotal = 0;
  let alertaDisparada = false;

  for (const obs of episodio) {
    const estado = estadoDeObservacion(obs, alertaDisparada);
    const accion = agente.seleccionarAccion(estado);

    recompensaTotal += calcularRecompensa(accion, obs.anomalia, alertaDisparada);

    const hayAnomalia = obs.anomalia != null;
    const esGrave = hayAnomalia && ANOMALIAS_GRAVES.includes(obs.anomalia);
    const esLeve  = hayAnomalia && ANOMALIAS_LEVES.includes(obs.anomalia);

    if (!hayAnomalia) {
      if (accion === ACCIONES.OBSERVAR) aciertos++;
      else falsoPositivos++;
    } else {
      const correcto =
        (accion === ACCIONES.ALARMA && esGrave) ||
        (accion === ACCIONES.AVISO && esLeve);
      if (correcto) aciertos++;
      else falsoNegativos++;
    }

    alertaDisparada = siguienteAlerta(alertaDisparada, accion, obs.anomalia);
  }

  // Restaurar epsilon.
  agente.epsilon = epsilonPrevio;

  return { aciertos, falsoPositivos, falsoNegativos, recompensaTotal };
}

// ===========================================================================
// UTILIDAD: media de un rango del historial
// ===========================================================================

/**
 * Media aritmética de un subarray [desde, hasta).
 *
 * @param {number[]} arr
 * @param {number} desde
 * @param {number} hasta
 * @returns {number}
 */
function media(arr, desde, hasta) {
  let suma = 0;
  for (let i = desde; i < hasta; i++) suma += arr[i];
  return suma / (hasta - desde);
}

// ===========================================================================
// 5) MAIN
// ===========================================================================

/**
 * Punto de entrada: crea el agente, lo entrena, lo evalúa e imprime resultados.
 */
function main() {
  const NUM_EPISODIOS = 1000;
  const NUM_EVAL = 10;

  console.log('=== Detección de anomalías en colmena (Q-Learning) ===\n');
  console.log(`Estados: ${NUM_ESTADOS} | Acciones: ${NUM_ACCIONES} | Pasos/episodio: ${PASOS_POR_EPISODIO}`);

  // --- Entrenamiento ---
  const agente = new QAgent(NUM_ESTADOS, NUM_ACCIONES);
  console.log(`\nEntrenando ${NUM_EPISODIOS} episodios...\n`);

  const { historialRecompensas } = entrenar(agente, NUM_EPISODIOS);

  // Historial resumido: media de recompensa cada 100 episodios.
  console.log('Historial de entrenamiento (media de recompensa por bloque):');
  const BLOQUE = 100;
  for (let desde = 0; desde < NUM_EPISODIOS; desde += BLOQUE) {
    const hasta = Math.min(desde + BLOQUE, NUM_EPISODIOS);
    const m = media(historialRecompensas, desde, hasta);
    console.log(`  episodios ${String(desde + 1).padStart(4)}-${String(hasta).padStart(4)}: ${m.toFixed(2)}`);
  }
  console.log(`\nEpsilon final: ${agente.epsilon.toFixed(4)}`);

  // --- Evaluación sobre episodios nuevos ---
  console.log(`\nEvaluando ${NUM_EVAL} episodios nuevos (explotación pura)...\n`);

  const totales = { aciertos: 0, falsoPositivos: 0, falsoNegativos: 0, recompensaTotal: 0 };

  for (let e = 0; e < NUM_EVAL; e++) {
    const episodio = crearEpisodio();
    const r = evaluarEpisodio(agente, episodio);

    totales.aciertos += r.aciertos;
    totales.falsoPositivos += r.falsoPositivos;
    totales.falsoNegativos += r.falsoNegativos;
    totales.recompensaTotal += r.recompensaTotal;

    console.log(
      `  Episodio ${String(e + 1).padStart(2)}: ` +
      `aciertos=${String(r.aciertos).padStart(3)} ` +
      `FP=${String(r.falsoPositivos).padStart(3)} ` +
      `FN=${String(r.falsoNegativos).padStart(3)} ` +
      `recompensa=${r.recompensaTotal.toFixed(2)}`
    );
  }

  const pasosTotales = NUM_EVAL * PASOS_POR_EPISODIO;
  console.log('\n--- Resumen de evaluación ---');
  console.log(`  Aciertos totales : ${totales.aciertos} / ${pasosTotales} (${(100 * totales.aciertos / pasosTotales).toFixed(1)}%)`);
  console.log(`  Falsos positivos : ${totales.falsoPositivos}`);
  console.log(`  Falsos negativos : ${totales.falsoNegativos}`);
  console.log(`  Recompensa media : ${(totales.recompensaTotal / NUM_EVAL).toFixed(2)} por episodio`);
}

// Ejecuta main() solo si el archivo se lanza directamente (no al importarlo).
if (require.main === module) {
  main();
}

// ===========================================================================
// EXPORTS
// ===========================================================================

module.exports = {
  entrenar,
  evaluarEpisodio,
  normalizarSensores,
  estadoDeObservacion,
  siguienteAlerta
};
