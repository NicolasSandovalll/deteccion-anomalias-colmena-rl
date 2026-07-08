/**
 * server.js
 * ---------------------------------------------------------------------------
 * Servidor Express que expone el entrenamiento del agente Q-Learning por HTTP.
 *
 * Sirve un frontend estático (index.html) y ofrece una API para lanzar un
 * entrenamiento completo y devolver sus métricas.
 * ---------------------------------------------------------------------------
 */

'use strict';

const path = require('path');
const express = require('express');
const cors = require('cors');

const { entrenar, evaluarEpisodio, estadoDeObservacion, siguienteAlerta } = require('./app');
const { crearEpisodio, PASOS_POR_EPISODIO } = require('./simulador');
const {
  QAgent,
  NUM_ESTADOS,
  NUM_ACCIONES,
  ACCIONES,
  ANOMALIAS_GRAVES,
  ANOMALIAS_LEVES,
  calcularRecompensa
} = require('./qlearning');

// ===========================================================================
// CONFIGURACIÓN EXPRESS
// ===========================================================================

const PUERTO = 3000;
const NUM_EPISODIOS_ENTRENO = 1000;
const NUM_EPISODIOS_EVAL = 10;
const NUM_EPISODIOS_SIMULACION = 100; // entrenamiento rápido para la demo en vivo

const servidor = express();

// CORS abierto para permitir peticiones desde el frontend.
servidor.use(cors());

// Sirve archivos estáticos (index.html, css, js) desde la carpeta actual.
servidor.use(express.static(path.join(__dirname)));

// ===========================================================================
// RUTAS
// ===========================================================================

/**
 * GET /api/train
 * Crea un agente nuevo, lo entrena 1000 episodios y lo evalúa sobre 10
 * episodios nuevos. Devuelve el historial de recompensas, las métricas de
 * evaluación agregadas y el tiempo de ejecución en segundos.
 */
servidor.get('/api/train', (req, res) => {
  const inicio = Date.now();

  // Entrenamiento.
  const agente = new QAgent(NUM_ESTADOS, NUM_ACCIONES);
  const { historialRecompensas } = entrenar(agente, NUM_EPISODIOS_ENTRENO);

  // Evaluación sobre episodios nuevos: se agregan los conteos de todos.
  const totales = { aciertos: 0, falsoPositivos: 0, falsoNegativos: 0, recompensaTotal: 0 };

  for (let e = 0; e < NUM_EPISODIOS_EVAL; e++) {
    const episodio = crearEpisodio();
    const r = evaluarEpisodio(agente, episodio);
    totales.aciertos += r.aciertos;
    totales.falsoPositivos += r.falsoPositivos;
    totales.falsoNegativos += r.falsoNegativos;
    totales.recompensaTotal += r.recompensaTotal;
  }

  const pasosTotales = NUM_EPISODIOS_EVAL * PASOS_POR_EPISODIO;
  const tiempoEjecucion = (Date.now() - inicio) / 1000; // segundos

  res.json({
    historialRecompensas,
    evaluacion: {
      aciertos: totales.aciertos,
      falsoPositivos: totales.falsoPositivos,
      falsoNegativos: totales.falsoNegativos,
      recompensaMedia: totales.recompensaTotal / NUM_EPISODIOS_EVAL,
      porcentajeAciertos: (100 * totales.aciertos) / pasosTotales
    },
    tiempoEjecucion
  });
});

/**
 * GET /api/simular
 * Entrena un agente rápido (100 episodios) y reproduce UN episodio nuevo paso
 * a paso en modo explotación (epsilon = 0), devolviendo el detalle de cada uno
 * de los 168 pasos para poder animar el comportamiento del agente en el frontend.
 */
servidor.get('/api/simular', (req, res) => {
  // Agente entrenado de forma rápida.
  const agente = new QAgent(NUM_ESTADOS, NUM_ACCIONES);
  entrenar(agente, NUM_EPISODIOS_SIMULACION);
  agente.epsilon = 0; // explotación pura durante la simulación

  const episodio = crearEpisodio();
  const pasos = [];

  let alertaDisparada = false;
  let totalAnomalias = 0;
  let totalAciertos = 0;
  let totalFalsoPositivos = 0;
  let totalFalsoNegativos = 0;

  for (const obs of episodio) {
    // Decisión del agente para este paso.
    const estado = estadoDeObservacion(obs, alertaDisparada);
    const accion = agente.seleccionarAccion(estado);
    const recompensa = calcularRecompensa(accion, obs.anomalia, alertaDisparada);

    // Clasificación del paso (misma lógica que evaluarEpisodio).
    const hayAnomalia = obs.anomalia != null;
    const esGrave = hayAnomalia && ANOMALIAS_GRAVES.includes(obs.anomalia);
    const esLeve = hayAnomalia && ANOMALIAS_LEVES.includes(obs.anomalia);

    if (hayAnomalia) totalAnomalias++;

    if (!hayAnomalia) {
      if (accion === ACCIONES.OBSERVAR) totalAciertos++;
      else totalFalsoPositivos++;
    } else {
      const correcto =
        (accion === ACCIONES.ALARMA && esGrave) ||
        (accion === ACCIONES.AVISO && esLeve);
      if (correcto) totalAciertos++;
      else totalFalsoNegativos++;
    }

    // Detalle del paso para el frontend (lecturas en valores crudos).
    pasos.push({
      paso: obs.paso,
      hora: obs.hora,
      lecturas: {
        temp: obs.lecturas.temp,
        humedad: obs.lecturas.humedad,
        vibracion: obs.lecturas.vibracion,
        flujo: obs.lecturas.flujo
      },
      anomalia: obs.anomalia,
      accion,
      recompensa
    });

    alertaDisparada = siguienteAlerta(alertaDisparada, accion, obs.anomalia);
  }

  res.json({
    pasos,
    totalPasos: pasos.length,
    totalAnomalias,
    totalAciertos,
    totalFalsoPositivos,
    totalFalsoNegativos
  });
});

/**
 * GET /api/health
 * Comprobación de vida del servidor.
 */
servidor.get('/api/health', (req, res) => {
  res.json({ status: 'ok' });
});

// ===========================================================================
// ARRANQUE
// ===========================================================================

servidor.listen(PUERTO, () => {
  console.log(`Servidor corriendo en http://localhost:${PUERTO}`);
});
