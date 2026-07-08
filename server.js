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

const { entrenar, evaluarEpisodio } = require('./app');
const { crearEpisodio, PASOS_POR_EPISODIO } = require('./simulador');
const { QAgent, NUM_ESTADOS, NUM_ACCIONES } = require('./qlearning');

// ===========================================================================
// CONFIGURACIÓN EXPRESS
// ===========================================================================

const PUERTO = 3000;
const NUM_EPISODIOS_ENTRENO = 1000;
const NUM_EPISODIOS_EVAL = 10;

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
