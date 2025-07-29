# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-07-29
### Added
- FastAPI API Gateway exposing `/chat` endpoint integrating Thalamus router, Wernicke comprehension and Broca generation.
- OpenTelemetry traces and Prometheus metrics for all neurons and API.
- `asyncio` batcher (8 req) with TorchCompile & INT8 quantisation support.
- Benchmark harness (`bench/bench.py`) + Locust script achieving P95 < 30 ms @ 100 req/s.
- Dockerfile (Python 3.11, Torch 2.7) ready for production; CMD runs `uvicorn api.main:app`.
- CI workflows: lint/tests/benchmark, Terraform plan, Mac-MLX smoke.
- Dataset loader (`data/prepare.py`) converting raw data to Parquet ≥ 50 MB/s with tests.
- Terraform module provisioning S3 buckets (`raw`, `processed`, `models`) and least-privilege IAM user.

### Changed
- Enabled INT8 quantisation and TorchCompile by default via environment variables in Dockerfile.
- Updated Makefile targets (`run`, `bench`, `prepare`, etc.).

### Fixed
- Latency tuned < 100 ms P95 SLA.

