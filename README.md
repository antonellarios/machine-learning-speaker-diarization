# 🎙️ Speaker Diarization — ResNet50 + VoxConverse

Sistema completo de diarización de hablantes (*"¿quién habla cuándo?"*) usando Deep Learning. Proyecto final académico de la materia Machine Learning — Tecnicatura Universitaria en Ciencia de Datos e Inteligencia Artificial Aplicada (UPATECO).

---

## 🎯 ¿Qué es la diarización de hablantes?

Es el proceso de identificar y segmentar automáticamente *quién habla en cada momento* dentro de una grabación de audio. Tiene aplicaciones en subtitulado automático, análisis de reuniones, call centers y sistemas de transcripción.

---

## 🧠 Arquitectura del sistema

```
Audio (.wav)
      │
      ▼
Espectrograma Mel (64 bandas · 16kHz)
      │
      ▼
ResNet50 adaptada (1 canal en lugar de 3)
      │
      ▼
Embedding de 512 dimensiones (huella de voz)
      │
      ▼
Clustering Jerárquico Aglomerativo (L2 normalizado)
      │
      ▼
Predicción RTTM: "spk_0 habla de 00:00 a 00:05"
      │
      ▼
Evaluación: DER / JER vs. Ground Truth
```

---

## 📊 Resultados de evaluación

Evaluado sobre **216 audios** del dataset VoxConverse (subset *dev*):

| Métrica | Resultado |
|---|---|
| **DER** (Diarization Error Rate) | **41.27%** |
| **JER** (Jaccard Error Rate) | **52.07%** |

### Desglose del error

| Componente | Tasa |
|---|---|
| Confusion Rate (identidad) | 40.98% |
| Miss Rate (voz no detectada) | 0.29% |
| False Alarm Rate (silencio como voz) | 0.00% |

**Interpretación:** El sistema detecta la presencia de voz con casi precisión perfecta (Miss 0.29%, FA 0.00%). El error proviene casi exclusivamente de la **confusión de identidad** — el modelo detecta correctamente que alguien habla, pero tiene dificultades para distinguir entre hablantes. Esto es esperable al usar una arquitectura de visión (ResNet) adaptada en lugar de una específica de audio como ECAPA-TDNN.

> En contexto académico, un DER < 50% con solo 3 epochs de entrenamiento sobre una GPU T4 valida que el pipeline funciona correctamente.

---

## 🔄 Pipeline completo (10 bloques)

| Bloque | Descripción |
|---|---|
| 1 | Configuración del entorno · instalación de PyAnnote + dependencias |
| 2 | Montaje de Google Drive · estructura de directorios persistente |
| 3 | Descarga del dataset VoxConverse (216 audios WAV + archivos RTTM) |
| 4 | EDA: duración · hablantes · silencios · overlap por archivo |
| 5 | Organización en lotes (batching) con checkpoints para reanudar |
| 6 | Dataset PyTorch · segmentación por RTTM · MelSpectrogram |
| 7 | Definición del modelo ResNet50 adaptado + Transfer Learning |
| 8 | Training loop · Precisión Mixta (AMP) · guardado automático |
| 9 | Inferencia · Clustering · generación de RTTM · cálculo de DER/JER |
| 10 | Testing con audios reales externos (MP3/WAV locales) |

---

## 🛠️ Decisiones técnicas destacadas

- **ResNet50 adaptada:** se modificó `conv1` de 3 canales (RGB) a 1 canal (espectrograma monocanal) para aplicar Transfer Learning desde ImageNet
- **Batch size 8:** reducido desde 16 para evitar CUDA Out Of Memory en GPU T4 con audios de duración variable
- **Precisión Mixta (AMP):** entrenamiento en float16/float32 mixto para reducir uso de VRAM y acelerar el entrenamiento
- **Checkpointing robusto:** el entrenamiento se puede interrumpir y reanudar exactamente desde el último epoch sin perder progreso
- **Segmentos mínimos de 0.5s:** filtro aplicado al parsear RTTM para evitar micro-segmentos que generan ruido en los embeddings
- **Normalización L2:** los embeddings se normalizan antes del clustering para usar distancia euclidiana como aproximación al coseno

---

## 🗂️ Dataset

- **VoxConverse** (subset *dev*) — 216 grabaciones de conversaciones reales en inglés
- Anotaciones en formato RTTM con timestamps y etiquetas de hablantes
- Fuente: [github.com/joonson/voxconverse](https://github.com/joonson/voxconverse)

---

## 🛠️ Stack tecnológico

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-ee4c2c?logo=pytorch)
![PyAnnote](https://img.shields.io/badge/PyAnnote-Audio-blueviolet)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Clustering-orange?logo=scikitlearn)

- **Framework:** PyTorch + torchaudio
- **Modelo base:** ResNet50 (torchvision) con Transfer Learning
- **Diarización:** PyAnnote Audio 3.1.1
- **Clustering:** AgglomerativeClustering (scikit-learn)
- **Métricas:** DiarizationErrorRate, JaccardErrorRate (pyannote.metrics)
- **Entorno:** Google Colab (GPU T4)

---

## 📁 Estructura del proyecto

```
machine-learning-speaker-diarization/
│
├── TP_FINAL_MACHINE_LEARNING_SPEAKER_DIARIZATION.ipynb
└── README.md
```

---

## 👩‍💻 Autora

**Antonella Ríos**
Junior Data Analyst | Python · SQL · Power BI · Machine Learning
[LinkedIn](https://www.linkedin.com/in/antonellarios) · [GitHub](https://github.com/antonellarios)
