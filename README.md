<!-- markdownlint-disable MD041 -->
# vLLM for Cornserve

A fork of [vLLM](https://github.com/vllm-project/vllm), maintained for use inside [Cornserve](https://github.com/cornserve-ai/cornserve) — a distributed inference platform for any-to-any multimodal AI.

## What's changed from upstream vLLM

- **Cornserve task executor.** Runs vLLM as a Cornserve task executor (fissioned component) rather than a standalone server.
- **Sidecar integration.** Sends and receives multimodal embeddings and hidden states through the Cornserve sidecar.
- **OTEL observability.** Propagates OpenTelemetry context across component boundaries and adds spans for scheduling and queuing.
- **Qwen3-Omni Talker.** Adds support for the Talker (and Vocoder fission) in Qwen3-Omni.

## Documentation

- vLLM: <https://docs.vllm.ai>
- Cornserve: <https://cornserve.ai>
