# ADR-0001: Arquitectura Modular de Neuron

Date: 2025-07-28

## Estado

Propuesto

## Contexto

Neuron debe construirse como un sistema de IA distribuido, evitando un modelo monolítico. Se requieren múltiples "neuronas" especializadas (Wernicke, Broca, etc.) coordinadas por un núcleo (Tálamo). Esto permite:

- Escalabilidad horizontal y vertical.
- Ahorro energético al activar solo módulos necesarios.
- Independencia de despliegue y versionado por componente.
- Facilidad de mantenimiento y evolución.

## Decisión

Adoptar una **arquitectura modular** con los siguientes principios:

1. Cada neurona es un micro-servicio autocontenido.
2. Comunicación gRPC/Protobuf inicial; posibilidad de cambiar a ZeroMQ si la latencia lo exige.
3. El Tálamo actúa como *router* MoE, expuesto vía FastAPI `/chat`.
4. Observabilidad estandarizada (OpenTelemetry + Prometheus) en todos los servicios.
5. Contenedores Docker reproducibles orquestados con `docker-compose` (futuro K8s).

## Consecuencias

- Mayor complejidad inicial por servicios múltiples.
- Facilita pruebas, CI/CD y despliegues independientes.
- Permite sustituir/actualizar neuronas sin downtime global.
- Requiere diseño robusto de protocolos y versionado de mensajes.

## Alternativas Consideradas

- LLM monolítico de >2 B parámetros: rechazado por consumo energético y falta de interpretabilidad.
- Enfoque serverless FaaS por neurona: descartado por latencia impredecible y coste.

## Acciones

- Definir esquemas Protobuf in `proto/` y versión 0.1 del bus.
- Implementar `TalamoRouter` stub con 3 expertos hard-coded.
- Configurar CI/CD y observabilidad comunes.

---

*Fin de ADR*