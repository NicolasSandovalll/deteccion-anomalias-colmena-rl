/**
 * qlearning.js
 * ---------------------------------------------------------------------------
 * Agente de Aprendizaje por Refuerzo (Q-Learning) para la detección de
 * anomalías en la colmena.
 *
 * Este módulo aporta tres piezas:
 *   1. La clase QAgent (tabla Q + política epsilon-greedy + actualización).
 *   2. discretizarEstado(...): convierte las lecturas continuas + contexto en
 *      un índice de estado entero y único.
 *   3. calcularRecompensa(...): función de recompensa que guía el aprendizaje.
 *
 * Consume los datos que produce simulador.js, pero no depende de él: solo
 * necesita recibir lecturas y etiquetas.
 * ---------------------------------------------------------------------------
 */

'use strict';

// ===========================================================================
// PARÁMETROS Y CONSTANTES GLOBALES
// ===========================================================================

/**
 * Hiperparámetros por defecto del agente.
 *   alpha        (α) : tasa de aprendizaje.
 *   gamma        (γ) : factor de descuento de recompensas futuras.
 *   epsilon      (ε) : probabilidad inicial de exploración.
 *   epsilonMin      : suelo de epsilon (nunca deja de explorar del todo).
 *   epsilonDecay    : factor multiplicativo de decaimiento de epsilon.
 */
const PARAMS_DEFECTO = {
  alpha: 0.1,
  gamma: 0.95,
  epsilon: 1.0,
  epsilonMin: 0.05,
  epsilonDecay: 0.995
};

/**
 * Espacio de acciones del agente.
 *   0 = observar (no hacer nada)
 *   1 = aviso    (alerta leve)
 *   2 = alarma   (alerta grave)
 */
const NUM_ACCIONES = 3;
const ACCIONES = { OBSERVAR: 0, AVISO: 1, ALARMA: 2 };

/**
 * Dimensiones del estado discretizado (radios del sistema mixto).
 *   temp, humedad, vibracion, flujo -> 3 niveles cada uno (3^4 = 81)
 *   hora                            -> 6 niveles
 *   alertaDisparada                 -> 2 niveles
 *
 * Total de estados = 3^4 * 6 * 2 = 972  (índices 0..971).
 */
const NIVELES = { temp: 3, humedad: 3, vibracion: 3, flujo: 3, hora: 6, alerta: 2 };
const NUM_ESTADOS =
  NIVELES.temp * NIVELES.humedad * NIVELES.vibracion *
  NIVELES.flujo * NIVELES.hora * NIVELES.alerta; // 972

/**
 * Clasificación de anomalías por gravedad.
 * Se usa en la función de recompensa para premiar la respuesta adecuada.
 */
const ANOMALIAS_GRAVES = ['avispas', 'despoblamiento', 'enjambrazon'];
const ANOMALIAS_LEVES  = ['lluvia', 'exceso_humedad'];

// ===========================================================================
// 1) CLASE QAgent
// ===========================================================================

class QAgent {
  /**
   * @param {number} numEstados  - tamaño del espacio de estados.
   * @param {number} numAcciones - tamaño del espacio de acciones.
   * @param {object} params      - hiperparámetros (se mezclan con PARAMS_DEFECTO).
   */
  constructor(numEstados, numAcciones, params = {}) {
    this.numEstados = numEstados;
    this.numAcciones = numAcciones;

    // Mezcla de parámetros: los pasados por el usuario pisan a los por defecto.
    const p = { ...PARAMS_DEFECTO, ...params };
    this.alpha = p.alpha;
    this.gamma = p.gamma;
    this.epsilon = p.epsilon;
    this.epsilonMin = p.epsilonMin;
    this.epsilonDecay = p.epsilonDecay;

    // Tabla Q: numEstados x numAcciones inicializada a ceros.
    this.Q = this.inicializarTablaQ();
  }

  /**
   * Crea la tabla Q como matriz numEstados x numAcciones rellena de ceros.
   *
   * @returns {number[][]}
   */
  inicializarTablaQ() {
    const tabla = new Array(this.numEstados);
    for (let s = 0; s < this.numEstados; s++) {
      tabla[s] = new Array(this.numAcciones).fill(0);
    }
    return tabla;
  }

  /**
   * Selecciona una acción para un estado dado usando la política epsilon-greedy:
   *   - Con probabilidad epsilon -> EXPLORA (acción aleatoria).
   *   - En caso contrario        -> EXPLOTA (mejor acción conocida = argmax Q).
   *
   * @param {number} estado - índice de estado (0..numEstados-1).
   * @returns {number} índice de acción.
   */
  seleccionarAccion(estado) {
    // Exploración.
    if (Math.random() < this.epsilon) {
      return Math.floor(Math.random() * this.numAcciones);
    }
    // Explotación: argmax sobre la fila del estado.
    return this._argmax(this.Q[estado]);
  }

  /**
   * Actualiza la tabla Q con la ecuación de Q-Learning:
   *
   *   Q(s,a) <- Q(s,a) + α · [ r + γ · max_a' Q(s',a') − Q(s,a) ]
   *
   * @param {number} estado          - estado actual s.
   * @param {number} accion          - acción tomada a.
   * @param {number} recompensa      - recompensa recibida r.
   * @param {number} estadoSiguiente - estado resultante s'.
   */
  actualizarQ(estado, accion, recompensa, estadoSiguiente) {
    const qActual = this.Q[estado][accion];
    const mejorFuturo = Math.max(...this.Q[estadoSiguiente]); // max_a' Q(s',a')

    // Diferencia temporal (TD target - valor actual).
    const objetivo = recompensa + this.gamma * mejorFuturo;
    const tdError = objetivo - qActual;

    this.Q[estado][accion] = qActual + this.alpha * tdError;
  }

  /**
   * Reduce epsilon multiplicándolo por epsilonDecay, sin bajar de epsilonMin.
   * Se suele llamar una vez por episodio para pasar de explorar a explotar.
   */
  decayEpsilon() {
    this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
  }

  /**
   * Devuelve el índice del valor máximo de un array.
   * Si hay empates, elige uno de ellos al azar para no sesgar siempre hacia la
   * primera acción (importante al inicio, cuando toda la fila vale 0).
   *
   * @param {number[]} fila
   * @returns {number}
   * @private
   */
  _argmax(fila) {
    let maxVal = -Infinity;
    let candidatos = [];

    for (let i = 0; i < fila.length; i++) {
      if (fila[i] > maxVal) {
        maxVal = fila[i];
        candidatos = [i];
      } else if (fila[i] === maxVal) {
        candidatos.push(i);
      }
    }
    return candidatos[Math.floor(Math.random() * candidatos.length)];
  }
}

// ===========================================================================
// 2) DISCRETIZACIÓN DEL ESTADO
// ===========================================================================

/**
 * Discretiza cada variable en su nivel correspondiente y las combina en un
 * único índice entero mediante un sistema numérico de base mixta.
 *
 * IMPORTANTE sobre las unidades:
 *   - temp y humedad se esperan en unidades físicas (°C y %).
 *   - vibracion y flujo se esperan NORMALIZADOS en [0, 1] (los umbrales 0.5/0.9
 *     y 0.3/0.7 así lo indican). Si vienen de simulador.js en escala cruda,
 *     hay que normalizarlos antes de llamar a esta función.
 *
 * Niveles:
 *   temp:      bajo (<34)   / normal (34-36) / alto (>36)      -> 0,1,2
 *   humedad:   seca (<55)   / normal (55-70) / húmeda (>70)    -> 0,1,2
 *   vibracion: baja (<0.5)  / normal (0.5-0.9) / alta (>0.9)   -> 0,1,2
 *   flujo:     bajo (<0.3)  / normal (0.3-0.7) / alto (>0.7)   -> 0,1,2
 *   hora:      0..5 (ya discretizada por bandas)
 *   alerta:    0 o 1
 *
 * @param {number} temp
 * @param {number} humedad
 * @param {number} vibracion   - normalizado [0,1]
 * @param {number} flujo       - normalizado [0,1]
 * @param {number} hora        - banda horaria 0..5
 * @param {number} alertaDisparada - 0 o 1
 * @returns {number} índice de estado en [0, NUM_ESTADOS-1] (0..971)
 */
function discretizarEstado(temp, humedad, vibracion, flujo, hora, alertaDisparada) {
  // --- Nivel de cada variable continua ---
  const nTemp      = temp      < 34   ? 0 : (temp      <= 36  ? 1 : 2);
  const nHumedad   = humedad   < 55   ? 0 : (humedad   <= 70  ? 1 : 2);
  const nVibracion = vibracion < 0.5  ? 0 : (vibracion <= 0.9 ? 1 : 2);
  const nFlujo     = flujo     < 0.3  ? 0 : (flujo     <= 0.7 ? 1 : 2);

  // --- Variables ya discretas (se saneen por seguridad) ---
  const nHora   = Math.min(NIVELES.hora - 1, Math.max(0, Math.floor(hora)));
  const nAlerta = alertaDisparada ? 1 : 0;

  // --- Composición en base mixta ---
  // El orden fija los "pesos" de cada dígito; da un índice único y compacto.
  let indice = nTemp;
  indice = indice * NIVELES.humedad   + nHumedad;
  indice = indice * NIVELES.vibracion + nVibracion;
  indice = indice * NIVELES.flujo     + nFlujo;
  indice = indice * NIVELES.hora      + nHora;
  indice = indice * NIVELES.alerta    + nAlerta;

  return indice; // 0..971
}

// ===========================================================================
// 3) FUNCIÓN DE RECOMPENSA
// ===========================================================================

/**
 * Calcula la recompensa de un paso a partir de la acción del agente, el tipo
 * de anomalía real (verdad de terreno) y si ya se había disparado una alerta.
 *
 * Acciones: 0=observar, 1=aviso, 2=alarma.
 *
 * Reglas (se acumulan; el costo por paso siempre aplica):
 *   -0.5 : siempre (costo por paso, incentiva no alarmar sin motivo).
 *   +50  : alarma (2) ante anomalía GRAVE y sin alerta previa (acierto fuerte).
 *   +25  : aviso  (1) ante anomalía LEVE  y sin alerta previa (acierto leve).
 *   -15  : cualquier alerta (acción>0) sin que haya anomalía (falsa alarma).
 *   -8   : mala clasificación:
 *            · alarma (2) ante anomalía leve, o
 *            · aviso  (1) ante anomalía grave.
 *   -5   : observar (0) habiendo una anomalía real (ignorarla).
 *
 * @param {number} accionAgente      - 0, 1 o 2.
 * @param {string|null} tipoAnomalia - nombre de la anomalía o null si no hay.
 * @param {boolean} alertaYaDisparada - true si ya se había alertado.
 * @returns {number} recompensa total del paso.
 */
function calcularRecompensa(accionAgente, tipoAnomalia, alertaYaDisparada) {
  const hayAnomalia = tipoAnomalia != null;
  const esGrave = hayAnomalia && ANOMALIAS_GRAVES.includes(tipoAnomalia);
  const esLeve  = hayAnomalia && ANOMALIAS_LEVES.includes(tipoAnomalia);

  // Costo por paso: siempre presente.
  let recompensa = -0.5;

  // Aciertos (solo si aún no se había disparado la alerta).
  if (accionAgente === ACCIONES.ALARMA && esGrave && !alertaYaDisparada) {
    recompensa += 50;
  }
  if (accionAgente === ACCIONES.AVISO && esLeve && !alertaYaDisparada) {
    recompensa += 25;
  }

  // Falsa alarma: alertar (aviso o alarma) sin anomalía real.
  if (accionAgente > ACCIONES.OBSERVAR && !hayAnomalia) {
    recompensa -= 15;
  }

  // Mala clasificación de la gravedad.
  if ((accionAgente === ACCIONES.ALARMA && esLeve) ||
      (accionAgente === ACCIONES.AVISO  && esGrave)) {
    recompensa -= 8;
  }

  // Ignorar una anomalía real.
  if (accionAgente === ACCIONES.OBSERVAR && hayAnomalia) {
    recompensa -= 5;
  }

  return recompensa;
}

// ===========================================================================
// EXPORTS
// ===========================================================================

module.exports = {
  QAgent,
  discretizarEstado,
  calcularRecompensa,

  // Constantes útiles para el entrenamiento / evaluación.
  PARAMS_DEFECTO,
  NUM_ACCIONES,
  NUM_ESTADOS,
  ACCIONES,
  ANOMALIAS_GRAVES,
  ANOMALIAS_LEVES
};
