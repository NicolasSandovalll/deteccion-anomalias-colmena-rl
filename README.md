# 🐝 Detección de anomalías en colmena · Q-Learning

Aplicación web interactiva que **enseña, explica y demuestra** el Aprendizaje por
Refuerzo (RL) aplicado a un problema de **detección de anomalías**. Un agente
entrenado con **Q-Learning** vigila una colmena a partir de las lecturas de sus
sensores y aprende a decidir, en cada momento, si debe observar, emitir un aviso
o disparar una alarma.

> Proyecto individual de la asignatura **Sistemas Inteligentes**.

---

## Índice

- [Descripción](#descripción)
- [El caso: detección de anomalías en una colmena](#el-caso-detección-de-anomalías-en-una-colmena)
- [Modelado en Aprendizaje por Refuerzo](#modelado-en-aprendizaje-por-refuerzo)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Ejecución](#ejecución)
- [Estructura del proyecto](#estructura-del-proyecto)
- [API del servidor](#api-del-servidor)
- [Detalles de implementación](#detalles-de-implementación)
- [Limitaciones](#limitaciones)
- [Referencias](#referencias)
- [Autoría y licencia](#autoría-y-licencia)

---

## Descripción

La aplicación combina **contenido educativo**, **visualizaciones** y una
**simulación interactiva** para mostrar cómo un agente de RL puede identificar
comportamientos anómalos. Todo el aprendizaje ocurre sobre datos sintéticos
generados por un simulador de colmena con ciclos circadianos, y el
comportamiento del agente puede observarse paso a paso en el navegador.

La web incluye:

- Introducción al problema y al Aprendizaje por Refuerzo.
- Un **diagrama del ciclo agente–entorno** etiquetado con el caso de la colmena.
- Los **conceptos fundamentales** (agente, entorno, estado, acción, recompensa,
  política) instanciados en este proyecto.
- La **explicación del algoritmo** Q-Learning (ecuación de actualización y
  estrategia *epsilon-greedy*).
- Una **demo interactiva**: entrenar al agente y ver su curva de recompensa, o
  reproducir un episodio completo paso a paso.
- **Conclusiones, limitaciones y posibles extensiones**.

---

## El caso: detección de anomalías en una colmena

Un monitor inteligente recibe lecturas de **4 sensores** (temperatura, humedad,
vibración y flujo de abejas) que siguen un **patrón circadiano** predecible a lo
largo del día. Sobre ese patrón normal pueden aparecer anomalías:

| Gravedad | Anomalía | Efecto en los sensores |
|----------|----------|------------------------|
| **Grave** | Avispas | Vibración ↑↑, flujo ↓↓ |
| **Grave** | Despoblamiento | Flujo ↓↓, vibración ↓ |
| **Grave** | Enjambrazón | Flujo ↑↑, vibración ↓↓ |
| **Leve** | Lluvia | Temperatura ↓, humedad ↑ |
| **Leve** | Exceso de humedad | Humedad ↑↑, temperatura ↓ ligero |

El objetivo del agente es responder de forma proporcionada: **alarma** ante las
graves, **aviso** ante las leves y **observar** cuando todo es normal.

---

## Modelado en Aprendizaje por Refuerzo

- **Agente**: el monitor inteligente que elige una acción en cada paso.
- **Entorno**: la colmena, que evoluciona y genera lecturas.
- **Estado**: las 4 lecturas discretizadas (3 niveles cada una) + la banda
  horaria (6) + un *flag* de alerta (2). En total **3⁴ × 6 × 2 = 972 estados**.
- **Acciones**: `0` observar · `1` aviso (leve) · `2` alarma (grave).
- **Recompensa**: `+50` alarma correcta · `+25` aviso correcto · `−0.5` por paso
  (coste) · `−15` falsa alarma · `−8` clasificación errónea · `−5` ignorar una
  anomalía real.
- **Algoritmo**: Q-Learning tabular con `α = 0.1`, `γ = 0.95` y `ε` que decae de
  `1.0` a `0.05` a lo largo del entrenamiento.

Un **episodio** equivale a una semana (7 días × 24 h = **168 pasos**). El agente
se entrena durante **1000 episodios** y se evalúa sobre episodios nuevos en modo
explotación pura (`ε = 0`), midiendo el **porcentaje de pasos clasificados
correctamente** (típicamente **> 93 %**).

---

## Requisitos

- [Node.js](https://nodejs.org/) **≥ 18** (probado con v22).
- npm (incluido con Node.js).

No requiere base de datos ni servicios externos: todo se ejecuta en local.

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone <URL-del-repositorio>
cd deteccion-anomalias-colmena-rl

# 2. Instalar dependencias (Express y CORS)
npm install
```

---

## Ejecución

### Aplicación web (recomendado)

```bash
npm start
```

Luego abre en el navegador: **<http://localhost:3000>**

Desde la web puedes:

- **Entrenar agente** → entrena 1000 episodios, evalúa 10 nuevos y dibuja la
  curva de recompensa junto con las métricas.
- **Ver simulación** → entrena un agente rápido y reproduce un episodio paso a
  paso, con la gráfica de sensores y la línea de tiempo de decisiones.

### Entrenamiento por consola

Para ejecutar el entrenamiento y la evaluación sin interfaz web:

```bash
npm run train
```

Imprime en la terminal el historial de recompensas por bloques y el resumen de
la evaluación.

---

## Estructura del proyecto

```
.
├── simulador.js     # Simulador de colmena: ciclos circadianos y anomalías
├── qlearning.js     # Agente Q-Learning: tabla Q, discretización y recompensa
├── app.js           # Orquestador: entrenamiento y evaluación (modo consola)
├── server.js        # Servidor Express que expone la API y sirve la web
├── index.html       # Interfaz web (contenido educativo + visualizaciones)
├── package.json     # Metadatos, scripts y dependencias
└── README.md
```

Responsabilidades:

- **`simulador.js`** genera episodios de lecturas (`crearEpisodio`) con patrón
  circadiano y anomalías inyectadas aleatoriamente.
- **`qlearning.js`** implementa la clase `QAgent`, la discretización del estado
  (`discretizarEstado`) y la función de recompensa (`calcularRecompensa`).
- **`app.js`** une simulador y agente: normaliza sensores, entrena
  (`entrenar`) y evalúa (`evaluarEpisodio`).
- **`server.js`** expone la API REST y sirve `index.html`.

---

## API del servidor

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/api/train` | Entrena 1000 episodios, evalúa 10 y devuelve el historial de recompensas + métricas + tiempo. |
| `GET` | `/api/simular` | Entrena un agente rápido y reproduce un episodio paso a paso (168 pasos con lecturas, acción y recompensa). |
| `GET` | `/api/health` | Comprobación de estado del servidor (`{ "status": "ok" }`). |

Ejemplo:

```bash
curl http://localhost:3000/api/health
# {"status":"ok"}
```

---

## Detalles de implementación

- **Discretización**: temperatura y humedad se discretizan en unidades físicas;
  vibración y flujo se **normalizan a [0, 1]** antes de discretizar. El estado
  se codifica como un índice único en base mixta (0–971).
- **Exploración vs explotación**: durante el entrenamiento el agente explora con
  probabilidad `ε` (que decae por episodio) y explota el resto del tiempo.
- **Número de estados**: el espacio real es **3⁴ × 6 × 2 = 972** estados; los
  índices van de 0 a 971 y la tabla Q es de 972 × 3.

---

## Limitaciones

- Q-Learning **tabular** no escala a espacios de estado grandes.
- El simulador inyecta **una anomalía por ventana**; en la realidad pueden
  solaparse varias.
- Los datos son **sintéticos**; el mundo real tiene más ruido y variabilidad.
- La función de recompensa se diseñó **manualmente**.
- El estado no incluye **historial temporal** (el agente no "recuerda" pasos
  anteriores).

Posibles extensiones: Deep Q-Networks (DQN), datos reales de colmenas, redes
recurrentes (LSTM) y ajuste de la función de recompensa.

---

## Referencias

1. Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An
   Introduction* (2ª ed.). MIT Press.
2. Watkins, C. J. C. H. & Dayan, P. (1992). *Q-learning*. Machine Learning,
   8(3–4), 279–292.
3. Mnih, V. et al. (2015). *Human-level control through deep reinforcement
   learning*. Nature, 518, 529–533.
4. Documentación de [Express](https://expressjs.com/) y
   [Chart.js](https://www.chartjs.org/docs/).

---

## Autoría y licencia

- **Autor**: Nicolás Sandoval Lagos.
- **Licencia**: MIT (ver `package.json`).
