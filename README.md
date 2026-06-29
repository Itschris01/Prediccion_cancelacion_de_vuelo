# Sistema de Predicción de Estados de Vuelos

Proyecto de ingeniería de datos y aprendizaje automático para predecir el estado operativo de vuelos comerciales en Estados Unidos.

El sistema clasifica cada vuelo programado en una de tres categorías:

* **0 — Puntual**
* **1 — Retrasado**
* **2 — Cancelado**

La solución implementa una arquitectura tipo **Medallion** sobre Databricks, utilizando las capas Bronze, Silver y Gold para transformar datos crudos de vuelos y variables meteorológicas en un conjunto de datos listo para entrenamiento de modelos de Machine Learning.

---

## Objetivo del proyecto

Desarrollar un pipeline analítico capaz de estimar si un vuelo será puntual, presentará un retraso superior a 15 minutos o será cancelado, a partir de variables disponibles en los registros operacionales programados, condiciones meteorológicas históricas, características temporales, congestión aeroportuaria y comportamiento histórico de rutas y aerolíneas.

El proyecto busca integrar procesamiento distribuido con Apache Spark, almacenamiento Delta Lake, ingeniería de características y modelos supervisados de clasificación multiclase.

---

## Arquitectura general

```text
Datos crudos de vuelos BTS
        │
        ▼
┌───────────────────────────────┐
│           BRONZE              │
│ Archivos CSV mensuales crudos │
│ almacenados en Volumes        │
└───────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│                   SILVER                 │
│ Tipado, limpieza, validación, eliminación│
│ de duplicados y tablas Delta             │
└──────────────────────────────────────────┘
        │
        ├───────────────► Catálogo geográfico de aeropuertos
        │
        ├───────────────► Datos meteorológicos históricos
        │
        ▼
┌─────────────────────────────────────────┐
│                    GOLD                 │
│ Features temporales, climáticas,        │
│ logísticas y de congestión aeroportuaria│
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│              MACHINE LEARNING            │
│ Random Forest, LightGBM, MLP y          │
│ Regresión Logística                      │
└─────────────────────────────────────────┘
```

---

## Fuente de datos

### Datos de vuelos

Los datos principales provienen del sistema de transporte aéreo de Estados Unidos, con registros mensuales de vuelos comerciales.

Las columnas base utilizadas incluyen:

```text
YEAR
MONTH
DAY_OF_MONTH
DAY_OF_WEEK
FL_DATE
OP_UNIQUE_CARRIER
OP_CARRIER_FL_NUM
ORIGIN_AIRPORT_ID
ORIGIN
DEST_AIRPORT_ID
DEST
CRS_DEP_TIME
DEP_TIME
DEP_DELAY
CRS_ARR_TIME
ARR_DELAY
CANCELLED
CANCELLATION_CODE
CRS_ELAPSED_TIME
DISTANCE
CARRIER_DELAY
WEATHER_DELAY
NAS_DELAY
SECURITY_DELAY
LATE_AIRCRAFT_DELAY
```

### Datos meteorológicos

El proyecto integra información meteorológica histórica por aeropuerto y fecha mediante Open-Meteo. Las variables incorporadas son:

```text
MaxTemperature
MinTemperature
Precipitation
Snowfall
MaxWindSpeed
```

También se utiliza un catálogo geográfico de aeropuertos para asociar códigos IATA con sus coordenadas de latitud y longitud.

---

## Estructura del proyecto

```text
.
├── bronze_a_silver
│   ├── Descompresión de archivos ZIP
│   ├── Lectura de archivos CSV
│   ├── Casting de tipos de datos
│   ├── Validación de calidad
│   ├── Eliminación de duplicados
│   └── Persistencia de tablas Delta Silver
│
├── 2_ingesta_clima
│   ├── Identificación de aeropuertos únicos
│   ├── Descarga de catálogo OpenFlights
│   ├── Asociación de aeropuertos y coordenadas
│   ├── Extracción de clima histórico
│   └── Persistencia de tablas meteorológicas Silver
│
├── 3_gold_feature_engineering
│   ├── Carga de vuelos y clima desde Unity Catalog
│   ├── Rescate de aeropuertos sin clima mediante KDTree
│   ├── Join entre vuelos y clima
│   ├── Imputación temporal forward/backward fill
│   ├── Variables cíclicas de calendario y hora
│   ├── Escalado de variables climáticas
│   ├── Métricas de congestión aeroportuaria
│   └── Creación de la tabla Gold final
│
├── 4_ml_classical_models
│   ├── Auditoría del dataset Gold
│   ├── Sanitización y validación de rangos
│   ├── Muestreo balanceado por clase
│   ├── Feature engineering adicional
│   ├── Target encoding de rutas y aerolíneas
│   ├── Random Forest
│   ├── LightGBM
│   ├── Multi-Layer Perceptron
│   ├── Regresión Logística
│   └── Persistencia de modelos con Joblib
│
└── README.md
```

---

## Tablas Delta generadas

| Capa   | Tabla                                              | Descripción                                                   |
| ------ | -------------------------------------------------- | ------------------------------------------------------------- |
| Silver | `workspace.default.silver_vuelos_rechazados`       | Registros con errores de calidad o campos críticos inválidos. |
| Silver | `workspace.default.silver_vuelos_cancelaciones`    | Vuelos limpios y tipados para análisis posterior.             |
| Silver | `workspace.default.silver_vuelos_retrasos`         | Vuelos no cancelados para análisis de retrasos.               |
| Silver | `workspace.default.silver_aeropuertos_coordenadas` | Aeropuertos con código IATA, latitud y longitud.              |
| Silver | `workspace.default.silver_clima_historico`         | Registros meteorológicos diarios por aeropuerto.              |
| Gold   | `workspace.default.gold_vuelos_features`           | Dataset final de variables predictoras y etiqueta multiclase. |

---

## Ingeniería de características

La capa Gold genera variables diseñadas para representar estacionalidad, clima, operación de vuelos y congestión de aeropuertos.

### Variables temporales cíclicas

```text
sin_month
cos_month
sin_dayofweek
cos_dayofweek
sin_dayofmonth
cos_dayofmonth
sin_hour
cos_hour
```

Estas transformaciones permiten que el modelo interprete correctamente relaciones circulares, por ejemplo, entre diciembre y enero o entre las 23:00 y las 00:00.

### Variables climáticas escaladas

```text
scaled_max_temp
scaled_min_temp
scaled_precipitation
scaled_snowfall
scaled_wind_speed
```

### Variables operacionales

```text
CRSElapsedTime
Distance
scaled_origin_volume
scaled_dest_volume
```

Las variables de volumen representan congestión aeroportuaria calculada por bloques horarios programados de salida y llegada.

### Variables derivadas en Machine Learning

```text
weather_traffic_origin
weather_traffic_dest
wind_distance
is_morning_rush
is_evening_rush
```

También se agregan variables categóricas de aerolínea mediante codificación one-hot y tasas históricas suavizadas de retraso y cancelación para rutas y aerolíneas.

---

## Definición de la variable objetivo

La etiqueta multiclase se construye de la siguiente forma:

```python
if Cancelled == 1:
    label = 2
elif DepDelay > 15:
    label = 1
else:
    label = 0
```

| Etiqueta | Clase     | Descripción                                        |
| -------- | --------- | -------------------------------------------------- |
| 0        | Puntual   | Vuelo sin retraso de salida superior a 15 minutos. |
| 1        | Retrasado | Vuelo con retraso de salida superior a 15 minutos. |
| 2        | Cancelado | Vuelo cancelado antes de completar la operación.   |

---

## Modelos entrenados

El proyecto compara cuatro enfoques de clasificación multiclase.

| Modelo              | Tipo                     | Propósito                                                     |
| ------------------- | ------------------------ | ------------------------------------------------------------- |
| Regresión Logística | Modelo lineal            | Baseline interpretable.                                       |
| Random Forest       | Ensamble Bagging         | Captura relaciones no lineales mediante múltiples árboles.    |
| LightGBM            | Gradient Boosting        | Modelo principal para maximizar desempeño en datos tabulares. |
| MLPClassifier       | Red neuronal feedforward | Benchmark de aprendizaje profundo.                            |

---

## Estrategia de entrenamiento

Para evitar que la clase mayoritaria domine el entrenamiento, se construye una muestra balanceada de:

```text
150,000 vuelos puntuales
150,000 vuelos retrasados
150,000 vuelos cancelados
```

Total:

```text
450,000 registros
```

La separación se realiza mediante partición estratificada:

```text
80% entrenamiento
20% prueba
```

El conjunto de prueba final contiene:

```text
90,000 vuelos
30,000 por cada clase
```

---

## Resultados obtenidos

Los resultados se evaluaron sobre un conjunto de prueba balanceado.

| Modelo              | Accuracy | Macro F1 |
| ------------------- | -------- | -------- |
| Regresión Logística | 0.48     | 0.48     |
| Random Forest       | 0.63     | 0.63     |
| MLP                 | 0.63     | 0.63     |
| **LightGBM**        | **0.64** | **0.64** |

### Resultados detallados del mejor modelo: LightGBM

| Clase     | Precision | Recall | F1-score |
| --------- | --------- | ------ | -------- |
| Puntual   | 0.62      | 0.58   | 0.60     |
| Retrasado | 0.54      | 0.59   | 0.56     |
| Cancelado | 0.77      | 0.76   | 0.76     |

LightGBM fue seleccionado como el modelo recomendado debido a su mejor desempeño general y a su capacidad para identificar vuelos cancelados con un F1-score de 0.76.

---

## Interpretación de resultados

El modelo identifica con mayor precisión los vuelos cancelados que los vuelos retrasados. Esto es consistente con la naturaleza del problema:

* Las cancelaciones suelen estar asociadas con condiciones climáticas severas, estacionalidad, aeropuerto, aerolínea y congestión.
* Los retrasos dependen también de factores operativos no disponibles en este dataset, como disponibilidad de tripulación, mantenimiento, puertas de embarque, retrasos del vuelo anterior y decisiones de control aéreo.
* La regresión logística obtuvo resultados inferiores, lo que indica que las relaciones entre las variables son principalmente no lineales.

---

## Persistencia de modelos

Los modelos y artefactos de preprocesamiento se almacenan mediante `joblib` en un archivo maestro.

```text
ecosistema_completo_vuelos_v2.joblib
```

El artefacto incluye:

```text
- Regresión Logística
- Random Forest
- LightGBM
- Red neuronal MLP
- Lista de features finales
- Columnas one-hot de aerolíneas
- Mapeos históricos de rutas
- Mapeos históricos de aerolíneas
- Metadatos de entrenamiento
- Clases objetivo
```

La ruta de persistencia utilizada dentro del volumen de Databricks es:

```text
/Volumes/workspace/default/bronze_vuelos/modelos/
```

---

## Tecnologías utilizadas

```text
Databricks Community Edition
Apache Spark
Spark SQL
Delta Lake
Unity Catalog
Python
Pandas
NumPy
SciPy
Scikit-learn
LightGBM
Joblib
Open-Meteo Historical Weather API
OpenFlights
```

---

## Requisitos principales

```bash
pip install pandas numpy scipy scikit-learn lightgbm joblib
```

En Databricks, PySpark y Delta Lake son proporcionados por el entorno de ejecución.

---

## Ejecución del proyecto

Los notebooks deben ejecutarse en el siguiente orden:

```text
1. bronze_a_silver
2. 2_ingesta_clima
3. 3_gold_feature_engineering
4. 4_ml_classical_models
```

Después de que las tablas Silver y Gold estén creadas, es posible volver a ejecutar únicamente el notebook de Machine Learning para probar ajustes de modelos sin descargar nuevamente los datos climáticos.

---

## Limitaciones

* El clima utilizado es histórico diario, no una predicción meteorológica disponible antes de cada vuelo.
* El modelo no incluye información sobre aeronave, vuelo anterior, mantenimiento, tripulación o disponibilidad de puertas.
* La evaluación actual usa una muestra balanceada, por lo que las métricas no representan directamente la distribución natural de cancelaciones en la operación real.
* Para un sistema productivo, se recomienda validar el modelo usando una división temporal, entrenando con meses pasados y evaluando con meses futuros.
* Para predicción operativa real, las variables climáticas históricas deberían sustituirse por pronósticos meteorológicos disponibles antes de la salida del vuelo.

---


