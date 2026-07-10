/**
 * simulador.js
 * ---------------------------------------------------------------------------
 * Simulador de colmena con ciclos circadianos y anomalías.
 *
 * El objetivo de este módulo es generar "episodios" sintéticos de lecturas de
 * sensores (temperatura, humedad, vibración y flujo de entrada/salida de
 * abejas) que sigan un patrón circadiano predecible a lo largo del día y que
 * puedan verse alterados por anomalías puntuales.
 * ---------------------------------------------------------------------------
 */

'use strict';

// ===========================================================================
// PARÁMETROS GLOBALES
// ===========================================================================

/**
 * Un episodio representa una semana completa de observaciones.
 * Con pasos de 1 hora: 7 días * 24 horas = 168 pasos.
 */
const PASOS_POR_EPISODIO = 168;

/**
 * El día se divide en 6 bandas de 4 horas cada una (0-5).
 * Cada banda tiene un "perfil" circadiano distinto de los sensores.
 *
 *   Banda 0 -> 00:00-03:59  (madrugada)
 *   Banda 1 -> 04:00-07:59  (amanecer)
 *   Banda 2 -> 08:00-11:59  (mañana)
 *   Banda 3 -> 12:00-15:59  (mediodía / pico de actividad)
 *   Banda 4 -> 16:00-19:59  (tarde)
 *   Banda 5 -> 20:00-23:59  (noche)
 */
const HORAS_POR_BANDA = 4;
const NUM_BANDAS = 6;

/**
 * Rangos plausibles de cada sensor. Se usan para dar contexto realista y para
 * "recortar" (clamp) los valores tras aplicar ruido o anomalías, de modo que
 * nunca salgan fuera de lo físicamente razonable.
 */
const RANGOS_SENSORES = {
  temp:      { min: 5,   max: 45,   unidad: '°C' },   // temperatura interior
  humedad:   { min: 20,  max: 100,  unidad: '%'  },   // humedad relativa
  vibracion: { min: 0,   max: 100,  unidad: 'ua' },   // nivel de vibración (unidad arbitraria)
  flujo:     { min: 0,   max: 200,  unidad: 'ab/min' } // abejas por minuto en la piquera
};

/**
 * Perfil circadiano base por banda horaria.
 * Para cada banda se define el valor MEDIO esperado de cada sensor.
 * El ruido Gaussiano se añade después sobre estos valores.
 *
 * Idea general del patrón diario:
 *   - Temperatura: la colmena se autorregula, pero sube algo al mediodía.
 *   - Humedad: más alta de madrugada/noche, más baja al mediodía.
 *   - Vibración: mínima de noche, máxima en horas de actividad.
 *   - Flujo: casi nulo de noche, máximo al mediodía (pecoreo).
 */
const PERFIL_CIRCADIANO = [
  /* Banda 0 - madrugada  */ { temp: 33.0, humedad: 70, vibracion: 10, flujo: 2   },
  /* Banda 1 - amanecer   */ { temp: 33.5, humedad: 65, vibracion: 25, flujo: 20  },
  /* Banda 2 - mañana     */ { temp: 34.5, humedad: 55, vibracion: 55, flujo: 90  },
  /* Banda 3 - mediodía   */ { temp: 35.5, humedad: 45, vibracion: 70, flujo: 140 },
  /* Banda 4 - tarde      */ { temp: 34.8, humedad: 50, vibracion: 50, flujo: 80  },
  /* Banda 5 - noche      */ { temp: 33.5, humedad: 62, vibracion: 20, flujo: 10  }
];

/**
 * Magnitud del ruido Gaussiano (desviación típica) para cada sensor.
 * Se mantiene "pequeño" para que el patrón circadiano siga siendo reconocible.
 */
const RUIDO_SENSORES = {
  temp:      0.4,
  humedad:   2.5,
  vibracion: 3.0,
  flujo:     6.0
};

/**
 * Catálogo de anomalías simulables y su rango de duración (en pasos).
 * La duración concreta de cada anomalía se sortea dentro de este rango.
 */
const TIPOS_ANOMALIA = ['avispas', 'despoblamiento', 'enjambrazon', 'lluvia', 'exceso_humedad'];

const DURACION_ANOMALIA = { min: 5, max: 25 }; // pasos (horas)

/**
 * Número de anomalías a inyectar por episodio (rango inclusivo).
 */
const ANOMALIAS_POR_EPISODIO = { min: 3, max: 4 };

// ===========================================================================
// UTILIDADES
// ===========================================================================

/**
 * Recorta un valor al rango [min, max] de un sensor dado.
 * Evita que el ruido o las anomalías produzcan valores imposibles.
 *
 * @param {number} valor
 * @param {string} sensor - clave dentro de RANGOS_SENSORES
 * @returns {number}
 */
function recortar(valor, sensor) {
  const { min, max } = RANGOS_SENSORES[sensor];
  return Math.min(max, Math.max(min, valor));
}

/**
 * Genera un número aleatorio con distribución Normal (media 0, sigma 1)
 * mediante el método de Box-Muller.
 *
 * @returns {number} muestra ~ N(0, 1)
 */
function gaussiana() {
  // Se evita el 0 exacto porque Math.log(0) = -Infinity.
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/**
 * Devuelve un entero aleatorio uniforme en el rango [min, max] (ambos incluidos).
 *
 * @param {number} min
 * @param {number} max
 * @returns {number}
 */
function enteroAleatorio(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

/**
 * Devuelve un elemento aleatorio de un array.
 *
 * @param {Array} arr
 * @returns {*}
 */
function elementoAleatorio(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

// ===========================================================================
// 1) HORA DEL DÍA -> BANDA HORARIA
// ===========================================================================

/**
 * Convierte el índice de paso del episodio en una de las 6 bandas horarias.
 *
 * Como cada paso equivale a 1 hora, la hora del día es (paso % 24) y la banda
 * se obtiene dividiendo esa hora entre HORAS_POR_BANDA (4).
 *
 * @param {number} paso - índice del paso dentro del episodio (0..167)
 * @returns {number} banda horaria en el rango 0..5
 */
function getHoraDelDia(paso) {
  const horaDelDia = ((paso % 24) + 24) % 24; // 0..23, robusto ante pasos negativos
  return Math.floor(horaDelDia / HORAS_POR_BANDA); // 0..5
}

// ===========================================================================
// 2) LECTURAS BASE SEGÚN EL CICLO CIRCADIANO
// ===========================================================================

/**
 * Genera las lecturas "sanas" (sin anomalía) para una banda horaria concreta.
 *
 * Toma el valor medio del perfil circadiano de esa banda y le suma ruido
 * Gaussiano pequeño e independiente por sensor. Después recorta cada lectura
 * a su rango físico válido.
 *
 * @param {number} hora - banda horaria (0..5) devuelta por getHoraDelDia
 * @returns {{temp:number, humedad:number, vibracion:number, flujo:number}}
 */
function generarLecturasBase(hora) {
  const perfil = PERFIL_CIRCADIANO[hora];

  return {
    temp:      recortar(perfil.temp      + gaussiana() * RUIDO_SENSORES.temp,      'temp'),
    humedad:   recortar(perfil.humedad   + gaussiana() * RUIDO_SENSORES.humedad,   'humedad'),
    vibracion: recortar(perfil.vibracion + gaussiana() * RUIDO_SENSORES.vibracion, 'vibracion'),
    flujo:     recortar(perfil.flujo     + gaussiana() * RUIDO_SENSORES.flujo,      'flujo')
  };
}

// ===========================================================================
// 3) INYECCIÓN DE ANOMALÍAS
// ===========================================================================

/**
 * Aplica el efecto de una anomalía sobre un conjunto de lecturas base.
 *
 * Cada anomalía desplaza uno o varios sensores respecto a su valor normal.
 * Las flechas describen la intensidad del efecto:
 *   ↑/↓  = cambio moderado
 *   ↑↑/↓↓ = cambio fuerte
 *
 *   - "avispas":        vibración ↑↑ (alarma), flujo ↓↓ (abejas no salen)
 *   - "despoblamiento": flujo ↓↓ (menos población), vibración ↓
 *   - "enjambrazon":    flujo ↑↑ (salida masiva), vibración ↓↓ (colmena queda vacía/tranquila)
 *   - "lluvia":         temp ↓, humedad ↑
 *   - "exceso_humedad": humedad ↑↑, temp ↓ ligero
 *
 * Devuelve un NUEVO objeto de lecturas (no muta el original), ya recortado a
 * los rangos válidos.
 *
 * @param {{temp:number, humedad:number, vibracion:number, flujo:number}} lecturas
 * @param {string} tipoAnomalia - una de TIPOS_ANOMALIA
 * @returns {{temp:number, humedad:number, vibracion:number, flujo:number}}
 */
function inyectarAnomalia(lecturas, tipoAnomalia) {
  // Copia para no modificar las lecturas base originales.
  const r = { ...lecturas };

  switch (tipoAnomalia) {
    case 'avispas':
      // Ataque de avispas: gran agitación y las abejas dejan de salir.
      r.vibracion += 35; // ↑↑
      r.flujo     -= 60; // ↓↓
      break;

    case 'despoblamiento':
      // Pérdida progresiva de población: sale muy poca abeja y baja la actividad.
      r.flujo     -= 55; // ↓↓
      r.vibracion -= 15; // ↓
      break;

    case 'enjambrazon':
      // Enjambrazón: salida masiva de abejas y luego colmena mucho más tranquila.
      r.flujo     += 70; // ↑↑
      r.vibracion -= 30; // ↓↓
      break;

    case 'lluvia':
      // Lluvia: baja la temperatura y sube la humedad ambiente.
      r.temp    -= 3;  // ↓
      r.humedad += 15; // ↑
      break;

    case 'exceso_humedad':
      // Exceso de humedad interior (mala ventilación): humedad muy alta.
      r.humedad += 25; // ↑↑
      r.temp    -= 1;  // ↓ ligero
      break;

    default:
      throw new Error(`Tipo de anomalía desconocido: "${tipoAnomalia}"`);
  }

  // Recorte final a rangos físicos válidos.
  r.temp      = recortar(r.temp,      'temp');
  r.humedad   = recortar(r.humedad,   'humedad');
  r.vibracion = recortar(r.vibracion, 'vibracion');
  r.flujo     = recortar(r.flujo,     'flujo');

  return r;
}

// ===========================================================================
// 4) CREACIÓN DE UN EPISODIO COMPLETO
// ===========================================================================

/**
 * Planifica las ventanas de anomalía de un episodio.
 *
 * Sortea entre ANOMALIAS_POR_EPISODIO.min y .max anomalías, cada una con un
 * tipo aleatorio, un paso de inicio y una duración (5..25 pasos). Se procura
 * que las ventanas NO se solapen para que cada paso tenga como mucho una
 * anomalía activa.
 *
 * @returns {Array<{tipo:string, inicio:number, fin:number}>}
 *          ventanas ordenadas por paso de inicio (fin es exclusivo)
 */
function planificarAnomalias() {
  const cuantas = enteroAleatorio(ANOMALIAS_POR_EPISODIO.min, ANOMALIAS_POR_EPISODIO.max);
  const ventanas = [];

  // Se intenta colocar cada anomalía sin solaparse; si tras varios intentos no
  // cabe, simplemente se descarta (para no entrar en bucles infinitos).
  const MAX_INTENTOS = 50;

  for (let i = 0; i < cuantas; i++) {
    let colocada = false;

    for (let intento = 0; intento < MAX_INTENTOS && !colocada; intento++) {
      const duracion = enteroAleatorio(DURACION_ANOMALIA.min, DURACION_ANOMALIA.max);
      const inicio   = enteroAleatorio(0, PASOS_POR_EPISODIO - duracion);
      const fin      = inicio + duracion; // exclusivo

      // ¿Se solapa con alguna ventana ya colocada?
      const solapa = ventanas.some(v => inicio < v.fin && fin > v.inicio);

      if (!solapa) {
        ventanas.push({ tipo: elementoAleatorio(TIPOS_ANOMALIA), inicio, fin });
        colocada = true;
      }
    }
  }

  // Orden cronológico para facilitar lectura/depuración.
  ventanas.sort((a, b) => a.inicio - b.inicio);
  return ventanas;
}

/**
 * Crea un episodio completo de PASOS_POR_EPISODIO (168) pasos.
 *
 * Para cada paso:
 *   1. Calcula la banda horaria.
 *   2. Genera las lecturas base circadianas con ruido.
 *   3. Si el paso cae dentro de una ventana de anomalía, la inyecta.
 *
 * Cada elemento del array resultante incluye información útil para el análisis
 * posterior (paso, hora/banda, lecturas y etiqueta de anomalía).
 *
 * @returns {Array<{
 *   paso:number,
 *   hora:number,
 *   lecturas:{temp:number, humedad:number, vibracion:number, flujo:number},
 *   anomalia: (string|null)
 * }>}
 */
function crearEpisodio() {
  const ventanas = planificarAnomalias();
  const episodio = [];

  for (let paso = 0; paso < PASOS_POR_EPISODIO; paso++) {
    const hora = getHoraDelDia(paso);
    let lecturas = generarLecturasBase(hora);

    // ¿Hay alguna anomalía activa en este paso? (fin exclusivo)
    const ventana = ventanas.find(v => paso >= v.inicio && paso < v.fin);
    const anomalia = ventana ? ventana.tipo : null;

    if (anomalia) {
      lecturas = inyectarAnomalia(lecturas, anomalia);
    }

    episodio.push({ paso, hora, lecturas, anomalia });
  }

  return episodio;
}

// ===========================================================================
// EXPORTS
// ===========================================================================

module.exports = {
  // Parámetros globales (por si el consumidor los necesita)
  PASOS_POR_EPISODIO,
  NUM_BANDAS,
  HORAS_POR_BANDA,
  RANGOS_SENSORES,
  PERFIL_CIRCADIANO,
  RUIDO_SENSORES,
  TIPOS_ANOMALIA,
  DURACION_ANOMALIA,
  ANOMALIAS_POR_EPISODIO,

  // Funciones principales
  getHoraDelDia,
  generarLecturasBase,
  inyectarAnomalia,
  crearEpisodio,

  // Utilidades (útiles para tests o reutilización)
  recortar,
  gaussiana,
  enteroAleatorio,
  elementoAleatorio,
  planificarAnomalias
};
