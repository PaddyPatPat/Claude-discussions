# Personal LLM Hosting Research

Research into GPU processing services for running large MoE (Mixture-of-Experts) LLM inference with privacy requirements.

## Requirements

- **Models**: MiniMax M2.1 (230B), GLM-4.7 (355B), Kimi K2.5 (1T)
- **Performance**: 25+ tokens/second
- **Memory**: 230-600GB (requires 4-8+ H100/H200 GPUs)
- **Privacy**: TEE with cryptographic attestation acceptable
- **Budget**: $500-2,000/month OpEx or $40,000 CapEx

## Key Findings

### Top Recommendations

| Rank | Service | TEE | Cost (100 hrs/month) |
|------|---------|-----|----------------------|
| 1 | Phala Network | Yes (8x H200) | ~$2,800 |
| 2 | io.net + Phala | Yes | ~$335-2,000 |
| 3 | Mac Studio Cluster | On-premise | $40K + $100/mo |

### TEE Landscape

- **Multi-GPU TEE NOW**: Phala, io.net (via Phala), Spheron
- **Single-GPU TEE**: Azure, Google Cloud (insufficient for large models)
- **Coming Q1 2026**: Akash AEP-65 confidential computing

### Hardware Contribution

- No viable Apple Silicon earning platforms exist
- Best credit offset: io.net (tokens offset inference costs)
- Best cash: Vast.ai ($500-1,500/month)

## Documents

- [GPU Services Research (Main)](gpu-services-research-2026.md) - Comprehensive analysis with comparison tables
- [Decentralized GPU Networks](../research/decentralized-gpu-networks-2026.md) - Akash, io.net, Render, Gensyn, Nosana, Golem
- [Decentralized Inference Research](../research/decentralized-gpu-inference-research-2026.md) - Flux, Spheron, Prime Intellect, Exabits, Bittensor, Ritual
- [Hardware Contribution Programs](../research/gpu-contribution-programs-2026.md) - Earning/offsetting costs with your hardware

## Watch List (2026)

| Timeline | Development | Impact |
|----------|-------------|--------|
| March 2026 | Akash AEP-65 TEE | Decentralized confidential computing |
| H1 2026 | GB200 NVL72 | 10x performance improvement |
| H2 2026 | NVIDIA Rubin | Rack-scale confidential computing |
