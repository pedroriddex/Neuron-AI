## 1. Entendimiento del Hito

Construir en **4–6 semanas** un MVP (Fase 0) de la plataforma **Neuron** compuesto únicamente por **Tálamo + Wernicke + Broca**, con **2–3 expertos**, **sin memoria persistente ni procesamiento emocional** y con una latencia **P95 < 100 ms**. El sistema debe:

- Exponer un endpoint **REST /chat** (FastAPI) con especificación OpenAPI.
- Cargar un corpus inicial (~10 GB) para pruebas.
- Emitir telemetría mediante **OpenTelemetry** y métricas Prometheus.
- Ejecutarse en contenedores reproducibles y pasar por CI/CD.

---

## 2. Suposiciones / Riesgos / Preguntas

- **Hardware**: disponibilidad de GPU ≥ 8 GB VRAM y un Mac Mini M2 con MLX para pruebas.
- **Datasets** limpios accesibles en `s3://neuron-data/raw/` antes del Día 3 del Sprint 1.
- **Riesgo‑R1** (precisión ↘ por int8): habilitar *fallback* fp16 vía flag.
- **Riesgo‑R2** (retraso ingestión datos): usar subset 1 GB para pruebas tempranas.
- **Riesgo‑R3** (overhead OTel): muestreo dinámico al 1 %.
- **Pregunta‑Q1** (resuelta): se confirma **FastAPI REST** como API inicial.

---

## 3. Estructura de Trabajo

### 3.1. Entregables → Épicas → Historias

- **E1 – Tálamo Core**
    - H1.1 Routing Expert Choice (3 expertos)
    - H1.2 Métricas de balance & latencia
- **E2 – Wernicke MVP**
    - H2.1 Carga modelo 350 M cuantizado
    - H2.2 Inferencia + tokenizer
- **E3 – Broca MVP**
    - H3.1 Carga modelo 450 M cuantizado
    - H3.2 Generación + post‑processing
- **E4 – Infra & Observabilidad**
    - H4.1 Docker + Compose base
    - H4.2 OpenTelemetry + Prometheus
- **E5 – Integración & Perf**
    - H5.1 API Gateway / FastAPI
    - H5.2 Benchmarks y profiling

### 3.2. Tareas Atómicas

| ID | Título | Descripción técnica (resumen) | Resultado / Criterios de aceptación | Artefactos | Depende de | Owner | Bloq. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T‑001 | Repo *neuron‑adrs* | Crear repo Git con plantilla ADR y ADR‑001 “Arquitectura modular” | ADR merged en `main` | `/adrs/ADR‑0001.md` | — | **Logan** | No |
| T‑002 | Docker base | `Dockerfile`Python 3.11 + Torch 2 + HF, push a registry | Imagen `neuron/base:0.1`ejecuta `import torch` OK | `Dockerfile` | — | Logan | Sí |
| T‑003 | Pipeline CI | GitHub Actions: lint, tests, build imagen | Jobs verdes en `main` | `.github/workflows/ci.yml` | T‑002 | Logan | Sí |
| T‑004 | Tálamo router stub | Clase `TalamoRouter` con 3 expertos hard‑coded | Unit test: elección experto < 5 ms | `talamo/router.py` | T‑002 | Logan | Sí |
| T‑005 | Telemetry router | Span OTel + histograma Prom en `route_input` | Métrica y span visibles | `talamo/telemetry.py` | T‑004 | Logan | No |
| T‑006 | Load Wernicke model | Script descarga + quantiza modelo 350 M; API `infer()` | `infer()` < 30 ms | `wernicke/model.py` | T‑002 | Logan | Sí |
| T‑007 | Tests Wernicke | PyTest; vector size 768; ≥ 95 % coverage | `tests/test_wernicke.py` | T‑006 | Logan | No |  |
| T‑008 | Load Broca model | Carga modelo 450 M + generación 50 tokens | Respuesta < 50 ms | `broca/model.py` | T‑002 | Logan | Sí |
| T‑009 | Tests Broca | PyTest; long ≤ 60 tokens; ≥ 95 % coverage | `tests/test_broca.py` | T‑008 | Logan | No |  |
| T‑010 | API Gateway FastAPI | Endpoint `/chat`, OpenAPI, integra Tálamo/Wernicke/Broca | `curl` respuesta JSON válida | `api/main.py` | T‑004,T‑006,T‑008 | Logan | Sí |
| T‑011 | Perf dataset loader | Ingestión 10 GB → Parquet con throughput ≥ 50 MB/s | `data/prepare.py` | — | Logan | No |  |
| T‑012 | Smoke E2E test | Docker‑compose: prompt→respuesta; test CI verde | `tests/e2e_smoke.py` | T‑010 | Logan | Sí |  |
| T‑013 | Benchmark harness | Script Locust/hey 100 req/s; métricas Prom | P95 < 100 ms | `bench/bench.py` | T‑012 | Logan | Sí |
| T‑014 | Latency tuning | TorchCompile + int8; batcher 8 req | P95 ↘ ≥ 20 % vs baseline | PR merged | T‑013 | Logan | Sí |
| T‑015 | Release 0.1 tag | SemVer, changelog, push images prod | Tag `v0.1.0` publicado | `CHANGELOG.md` | T‑014 | Logan | No |
| T‑016 | CI runner Metal/MLX | Configurar Mac Mini, smoke fp16/int8 | Runner acepta job ejemplo | `.github/runners/mac` | — | Logan | No |
| T‑017 | Buckets S3 + IAM | Crear `raw/processed/models/`; políticas least‑priv | Acceso S3 validado | `infra/terraform/s3.tf` | — | Logan | Sí |

### 3.3. Orden / Camino Crítico

1. T‑001 → T‑002 → T‑003
2. T‑004 & T‑006 & T‑008 (paralelo) → T‑005, T‑007, T‑009
3. T‑010 → T‑012 → T‑013 → T‑014 → T‑015
4. Paralelo: T‑011, T‑016, T‑017

### 3.4. Riesgos & Mitigaciones

| Riesgo | Prob. | Impacto | Mitigación |
| --- | --- | --- | --- |
| R‑CPU‑Spike (precisión ↘) | M | M | Flag `--precision fp16` de *fallback* |
| R‑Dataset‑Lag | M | L | Subset 1 GB para pruebas tempranas |
| R‑Trace‑Overhead | L | M | Muestreo 1 % y ajuste dinámico |

---

## 4. Definition of Done (Fase 0)

- Endpoint `/chat` genera respuesta coherente **P95 < 100 ms**.
- Cobertura tests unitarios ≥ 90 % en Tálamo, Wernicke, Broca.
- Imagen Docker reproducible publicada en registry interno.
- Traces OTel visibles y métricas Prometheus en dashboard.
- Harness de rendimiento almacena resultados histórico.
- Al menos 3 ADRs aprobados en `neuron‑adrs`.

---

## 5. Próximos Pasos

1. **Kick‑off Sprint 1** (30 min) – Blanca convoca.
2. Logan comienza **T‑001** y solicita acceso al registry y a `neuron‑data`.
3. Pedro provisiona GPU runner y Mac Mini CI.
4. Blanca crea tablero Jira con IDs y dependencias.

¡Bienvenido al mando, Logan! 🚀