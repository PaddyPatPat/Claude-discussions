# Decentralized GPU Networks for LLM Inference - Research Report

**Research Date:** January-February 2026
**Use Case:** Large MoE models (230B-1T parameters, 230-600GB memory) at 25+ tokens/second with TEE privacy requirements

---

## Executive Summary

| Service | Status | Large Model Support | TEE Support | Best For |
|---------|--------|---------------------|-------------|----------|
| **Spheron Network** | Live (Mainnet) | Yes (via bare metal H100 clusters) | H100 CC Available | Production workloads, cost-sensitive deployments |
| **Prime Intellect** | Live (Inference Preview) | Yes (pipeline parallelism across nodes) | No (TOPLOC verification only) | Distributed inference across consumer GPUs |
| **Bittensor** | Live (Multiple Subnets) | Partial (depends on subnet) | Roadmap (TEE integration planned) | Decentralized AI marketplace |
| **Flux/FluxEdge** | Live (Mainnet) | Limited (primarily consumer GPUs) | No | General compute, smaller models |
| **Exabits** | Live (Early Stage) | Yes (H100/H200 available) | Roadmap (NEAR partnership) | GPU tokenization, institutional investors |
| **Ritual** | Devnet (Infernet Live) | Planned | Yes (TEE + ZKP architecture) | On-chain AI, smart contract integration |

---

## 1. Flux Network (RunOnFlux / FluxEdge)

### Current Status
**Operational Status:** Live Mainnet (since 2018)
**Network Scale:** 10,000+ nodes across 66+ countries, 560+ independent operators
**Applications Deployed:** 25,000+ applications across 75+ countries

### GPU Types Available
- **Consumer GPUs:** RTX 3090, RTX 4090 (primary offering)
- **Datacenter GPUs:** Via FluxEdge Premium tier (H100, A100, Blackwell) through partner Hyperstack
- **Mix:** Predominantly consumer-grade hardware from decentralized node operators

### Pricing
- **Cost Savings:** Claims up to 90% lower cost than AWS/GCP
- **FluxEdge Premium:** Requires KYC1 verification for enterprise GPUs
- **Specific rates:** Not publicly published; marketplace-based pricing

### TEE / Confidential Computing
- **Status:** NOT AVAILABLE
- **Roadmap:** No announced plans for TEE integration

### Large Model Inference Support (500GB+)
- **Verdict:** NOT SUITABLE
- **Reason:** Network primarily consists of consumer-grade nodes with limited VRAM
- FluxEdge Premium tier offers H100s but not designed for multi-node tensor parallelism

### Tensor Parallelism
- **Status:** NOT SUPPORTED natively
- **Architecture:** Designed for independent application deployment, not distributed inference

### Data Privacy Architecture
- **Model:** Decentralized, censorship-resistant infrastructure
- **Guarantees:** No formal privacy guarantees for AI inference
- **Concern:** Data processed on untrusted consumer nodes

### Network Reliability
- **Uptime:** Not formally guaranteed
- **Redundancy:** Geographic distribution provides natural redundancy

### Sources
- [Flux Official Site](https://runonflux.com/)
- [FluxCloud](https://runonflux.com/fluxcloud/)
- [FluxNodes Documentation](https://docs.runonflux.com/flux/fluxnodes/what-are-fluxnodes)
- [Fluence Decentralized Cloud Guide](https://www.fluence.network/blog/decentralized-cloud-computing-guide/)

---

## 2. Spheron Network

### Current Status
**Operational Status:** Live Mainnet
**Revenue:** $12M+ ARR (as of August 2025)
**Token Launch:** July 25, 2025 (SPON token)

### GPU Types Available
| GPU Type | Availability | Memory |
|----------|--------------|--------|
| NVIDIA B200 SXM6 | On-demand | 192GB HBM3e |
| NVIDIA H200 | On-demand | 141GB HBM3e |
| NVIDIA H100 SXM5 | On-demand | 80GB HBM3 |
| NVIDIA A100 PCIE | On-demand | 80GB HBM2e |

### Pricing (Per GPU-Hour)
| GPU | Spot | On-Demand | Reserved (12mo) |
|-----|------|-----------|-----------------|
| H100 | ~$0.72+ | Market rate | Discounted |
| H200 | $1.56 | Higher | $1.56 base |
| B200 SXM6 | $1.21 | $4.71 | $3.20 |
| 8x H100 Bare Metal | - | $16.56/hr | - |
| 8x A100 Bare Metal | - | $13.05/hr | - |

**Cost Advantage:** Claims up to 47x lower than Google/Amazon for comparable GPU resources

### TEE / Confidential Computing
- **Status:** AVAILABLE (via NVIDIA H100/H200 Confidential Computing)
- **Implementation:** Hardware-based TEE with CVM (Confidential Virtual Machine)
- **Features:**
  - Device attestation verifies GPU authenticity
  - Firmware integrity verification
  - Protection from hypervisor, cloud provider, and physical access
- **GPU Support:** H100 and H200 with CC-On mode

### Large Model Inference Support (500GB+)
- **Verdict:** CAPABLE with bare metal clusters
- **Configuration:** 8x H100 SXM5 = 640GB VRAM per node
- **Multi-node:** Possible with bare metal rental, user must configure

### Tensor Parallelism
- **Status:** User-managed (not platform-native)
- **Infrastructure:** Bare metal servers with NVLink connectivity within node
- **Multi-node TP:** Requires user setup; relies on standard frameworks (vLLM, TensorRT-LLM)

### Data Privacy Architecture
- **Dual-node model:**
  - Standard Provider Nodes: Enterprise data center resources
  - Fizz Nodes: Edge/consumer GPU contributions
- **TEE Guarantees:** Hardware isolation blocks unauthorized access during execution
- **Compliance:** Suitable for healthcare, finance, government (with CC-enabled GPUs)

### Network Reliability
- **SLA:** 99% uptime claimed
- **Infrastructure:** Tier 3/4 data centers
- **Pricing Model:** All-inclusive (no hidden fees for CPU, memory, storage, idle time)

### Assessment for Your Use Case
- **Feasibility:** HIGH for 230-600GB models
- **Configuration:** 8x H100 bare metal ($16.56/hr) provides 640GB VRAM
- **TEE:** Available on H100/H200
- **Limitation:** 25+ tok/sec may require optimization; no native multi-node orchestration

### Sources
- [Spheron Official](https://www.spheron.network/)
- [Spheron Docs](https://docs.spheron.network/)
- [Spheron H200 Rental](https://www.spheron.network/gpu-rental/h200/)
- [Spheron Confidential Computing Blog](https://blog.spheron.network/maximize-security-using-nvidia-confidential-computing)
- [Messari Spheron Report](https://messari.io/report/understanding-spheron-a-comprehensive-overview)

---

## 3. Prime Intellect

### Current Status
**Operational Status:** Live (Inference in Preview)
**Funding:** $15M raised for peer-to-peer AI protocol
**Key Releases:**
- INTELLECT-1 (10B, Oct 2024)
- INTELLECT-2 (32B, 2025)
- INTELLECT-3 (100B+ MoE, 2025)

### GPU Types Available
- **Consumer GPUs:** RTX 4090, RTX 3090 (primary focus)
- **Datacenter GPUs:** H100, A100 (via Akash, io.net, Vast.ai, Lambda integration)
- **Philosophy:** Built for heterogeneous hardware including consumer-grade

### Pricing
- **Compute Marketplace:** Aggregates pricing from Akash, io.net, Vast.ai, Lambda
- **Inference API:** OpenAI-compatible API; pricing varies by model (details forthcoming)
- **Value Prop:** Best rental prices through aggregation

### TEE / Confidential Computing
- **Status:** NOT AVAILABLE
- **Alternative:** TOPLOC verification system
  - Locality-sensitive hashing for activation verification
  - Detects unauthorized model/prompt/precision modifications
  - 0.000925% false positive rate in production
  - 25x cheaper than re-executing inference for verification
- **Roadmap:** No announced TEE plans

### Large Model Inference Support (500GB+)
- **Verdict:** DESIGNED FOR THIS USE CASE
- **Architecture:** Pipeline parallelism across geographically distributed nodes
- **Proven:** DeepSeek-R1 scale models supported
- **Implementation:** PRIME-VLLM (pipeline-parallel vLLM over public networks)

### Tensor Parallelism
- **Pipeline Parallelism:** Primary approach for decentralized inference
  - Model divided into sequential stages
  - Each device stores only its stage
  - Hidden states streamed between peers
  - Works across H100s and RTX 4090s
- **Limitation:** Pipeline bubbles cause GPU idle time (active area of research)

### Data Privacy Architecture
- **Model:** Permissionless infrastructure
- **Verification:** TOPLOC provides computational integrity (not privacy)
- **Privacy:** Limited - data flows between untrusted nodes
- **Security:** Docker image deployment for environment control

### Network Reliability
- **Latency Target:** Engineered for 100ms public internet latencies
- **Validation:** Adversarial validation through TOPLOC
- **Infrastructure:** PRIME-IROH custom P2P communication library

### Assessment for Your Use Case
- **Feasibility:** MODERATE
- **Strengths:**
  - Best-in-class distributed inference for very large models
  - Can aggregate consumer + datacenter GPUs
  - Pipeline parallelism specifically designed for your scale
- **Weaknesses:**
  - No TEE support (TOPLOC is verification, not privacy)
  - 25+ tok/sec challenging over public internet
  - Privacy requirements cannot be met

### Sources
- [Prime Intellect Official](https://www.primeintellect.ai/)
- [Inference Overview Docs](https://docs.primeintellect.ai/inference/overview)
- [SYNTHETIC-2 Blog](https://www.primeintellect.ai/blog/synthetic-2)
- [Decentralized Inference Blog](https://www.primeintellect.ai/blog/inference)
- [Akash Integration](https://akash.network/blog/prime-intellect-integrates-permissionless-akash-gpus/)
- [Prime-VLLM GitHub](https://github.com/PrimeIntellect-ai/prime-vllm)

---

## 4. Exabits

### Current Status
**Operational Status:** Live (Early Stage)
**Founded:** 2021
**Institutional Backing:** Harvard Innovation Labs member
**Recent Partnerships:**
- NEAR Foundation Grant (April 2025)
- GamerHash AI integration (December 2024)

### GPU Types Available
| Category | GPUs |
|----------|------|
| Datacenter | GB200, H200, H100, A100 |
| Consumer | RTX 5090, RTX 4090 |
| AMD | MI50 |

### Pricing
- **Model:** Tokenized GPU marketplace
- **Specific Rates:** Not publicly published
- **Value Prop:** Low-cost compute through tokenization

### TEE / Confidential Computing
- **Status:** ROADMAP
- **NEAR Partnership:** Working on "privacy and verifiability" for decentralized AI
- **Timeline:** No specific dates announced

### Large Model Inference Support (500GB+)
- **Verdict:** POTENTIALLY CAPABLE
- **Hardware Available:** H100, H200, GB200 sufficient for large models
- **Limitation:** Platform maturity unclear; no proven large-scale deployments documented

### Tensor Parallelism
- **Status:** UNKNOWN
- **Documentation:** Limited technical documentation available

### Data Privacy Architecture
- **Focus Areas:** Privacy and verifiability (per NEAR partnership)
- **Current State:** No detailed privacy architecture documented
- **Target Market:** Web2 enterprises and Web3 protocols

### Network Reliability
- **Infrastructure:** Data centers + individual actors
- **SLA:** Not published

### Assessment for Your Use Case
- **Feasibility:** UNCERTAIN
- **Strengths:**
  - Has datacenter-grade GPUs (H100/H200/GB200)
  - Privacy focus in roadmap
- **Weaknesses:**
  - Platform maturity questionable
  - No TEE available today
  - Limited technical documentation
  - No proven large model inference deployments

### Sources
- [Exabits Official](https://www.exabits.ai/)
- [Hyperbolic GPU Landscape](https://www.hyperbolic.ai/blog/gpu-marketplace-landscape)
- [NEAR Partnership Announcement](https://www.globenewswire.com/news-release/2025/04/28/3069524/0/en/Exabits-teams-up-with-NEAR-to-push-the-boundaries-of-decentralized-AI.html)
- [Fluence GPU Marketplaces](https://www.fluence.network/blog/best-gpu-rental-marketplaces/)

---

## 5. Bittensor

### Current Status
**Operational Status:** Live Mainnet
**Network Scale:** 120+ active subnets (dramatic growth from 32 pre-dTAO)
**Key Event:** First halving approaching (2026)

### Relevant Subnets for Inference

| Subnet | Focus | Status |
|--------|-------|--------|
| **Chutes (SN64)** | Serverless AI inference | Live, 100K+ users, 30B tokens/day |
| **Nineteen (SN19)** | Ultra-low-latency inference | Live |
| **Targon (SN4)** | Confidential computing, AI verification | Live, ~$10.4M projected ARR |
| **Omron (SN2)** | ZK Proof of Inference | Live |
| **Templar (SN3)** | Distributed training | Live |
| **Celium** | GPU rental marketplace | Live |

### GPU Types Available
- **Varies by Subnet:** Each subnet has different provider requirements
- **Chutes (SN64):** CUDA 12.2-12.6 required, Docker-based
- **Celium:** H100, H200 available for rental
- **General:** Mix of consumer and datacenter GPUs across network

### Pricing
- **Chutes:** Claims ~85% lower cost than AWS for comparable inference
- **Model:** Marketplace-based; miners compete for emissions
- **Payment:** TAO tokens

### TEE / Confidential Computing
- **Targon (SN4):** Confidential computing platform for secure inference-as-a-service
- **Chutes:** TEE integration "aiming to integrate" (not live)
- **Omron (SN2):** ZK Proof of Inference (verification, not privacy)
- **Status:** PARTIAL - Targon offers confidential computing

### Large Model Inference Support (500GB+)
- **Verdict:** CHALLENGING
- **SN19 DSIS:** "Decentralized Subnet Inference at Scale" architecture
- **Limitation:** Consumer GPU focus limits single-provider capacity
- **Aggregation:** Would require coordinated multi-miner setup

### Tensor Parallelism
- **Status:** NOT NATIVE
- **Architecture:** Individual miners process requests
- **DSIS (SN19):** May support distributed inference but documentation limited

### Data Privacy Architecture
- **Anonymity:** Market actors are anonymous endpoints
- **Output Filtering:** Can filter sensitive information locally
- **Targon (SN4):** Enterprise-grade privacy/security guarantees
- **ZK (SN2):** Verifiable inference without revealing inputs

### Network Reliability
- **Validation:** Adversarial validation (multiple miners cross-verify)
- **Uptime:** Network-wide redundancy
- **SLA:** No formal guarantees

### Assessment for Your Use Case
- **Feasibility:** LOW-MODERATE
- **Best Option:** Targon (SN4) for confidential computing needs
- **Strengths:**
  - Large decentralized network
  - Privacy-focused subnets exist
  - Cost-effective for smaller models
- **Weaknesses:**
  - 500GB+ model inference not well-suited to architecture
  - TEE only in specific subnet (Targon)
  - No native tensor parallelism
  - 25+ tok/sec challenging for large models

### Sources
- [Bittensor Docs](https://docs.bittensor.com/)
- [Ultimate Guide to Bittensor 2026](https://www.tao.media/the-ultimate-guide-to-bittensor-2026/)
- [Chutes Subnet](https://learnbittensor.org/subnets/64)
- [Subnet Alpha](https://subnetalpha.ai/)
- [Grayscale Bittensor Research](https://research.grayscale.com/reports/bittensor-on-the-eve-of-the-first-halving-research)
- [Omron ZK Inference](https://medium.com/@tensorplexlabs/bittensor-subnet-2-omron-zero-knowledge-machine-learning-4dfa7f192fcd)

---

## 6. Ritual

### Current Status
**Operational Status:** Devnet (Infernet Live on mainnet)
**Funding:** $25M Series A (June 2024) from Archetype, Accel, Robot Ventures, Polychain
**Founded:** 2023, New York

### Components Status
| Component | Status |
|-----------|--------|
| Infernet | **Live** on multiple EVM chains |
| Ritual Chain | Private Testnet |
| Cascade (privacy) | In Development |

### GPU Types Available
- **Current (Infernet):** GPU optional; CUDA-enabled GPU recommended for advanced setups
- **Planned (Ritual Chain):** Specialized nodes for GPU-based AI inference

### Pricing
- **Current:** Not publicly documented
- **Model:** Will support heterogeneous compute pricing

### TEE / Confidential Computing
- **Status:** CORE ARCHITECTURE FEATURE
- **Implementation:**
  - Intel SGX support
  - AWS Nitro Enclaves support
  - Native TEE execution nodes planned for Ritual Chain
- **Cascade:** Proprietary privacy solution in development
- **ZKPs:** On-chain verification of off-chain computations without revealing inputs

### Large Model Inference Support (500GB+)
- **Verdict:** NOT AVAILABLE YET
- **Planned:** Ritual Chain designed for heterogeneous workloads including large AI inference
- **Current:** Infernet focused on smart contract integration, not large-scale inference

### Tensor Parallelism
- **Status:** NOT DOCUMENTED
- **Architecture:** Designed for on-chain AI calls, not distributed large model inference

### Data Privacy Architecture
- **Multi-layer approach:**
  1. TEEs (Intel SGX, Nitro Enclaves) for secure execution
  2. ZKPs for computation verification without input revelation
  3. Cascade (planned) for native privacy
- **Guarantees:** Computational integrity and privacy through cryptographic infrastructure

### Network Reliability
- **Infernet:** Lightweight oracle network connecting off-chain compute to on-chain
- **Ritual Chain:** Distributed verification via Symphony protocol
- **Node Specialization:** Operators choose TEE, ZK, or GPU specialization

### Assessment for Your Use Case
- **Feasibility:** NOT READY
- **Strengths:**
  - Best-in-class privacy architecture (TEE + ZKP)
  - Strong institutional backing
  - Purpose-built for secure AI compute
- **Weaknesses:**
  - Mainnet not launched
  - Large model inference not current focus
  - Timeline uncertain

### Sources
- [Ritual Official](https://ritual.net/blog/introducing-ritual)
- [Ritual Foundation Docs](https://www.ritualfoundation.org/docs/landscape/ritual-vs-other-crypto-x-ai)
- [Ritual Academy FAQs](https://ritual.academy/ritual/faqs/)
- [Gate.com Ritual Guide](https://www.gate.com/learn/articles/a-simple-guide-to-ritual-the-open-ai-infrastructure-network/4594)
- [Mitosis Ritual Overview](https://university.mitosis.org/inside-ritual-network-the-architecture-use-cases-and-community-powering-ritualnet/)

---

## Comparative Analysis for Your Requirements

### Requirement: 230B-1T Parameters (230-600GB Memory)

| Service | Capability | Notes |
|---------|-----------|-------|
| Spheron | **YES** | 8x H100 bare metal = 640GB VRAM |
| Prime Intellect | **YES** | Pipeline parallelism across nodes |
| Exabits | **MAYBE** | Has hardware, unproven platform |
| Bittensor | **PARTIAL** | Requires custom multi-miner setup |
| Flux | **NO** | Consumer GPU focus |
| Ritual | **FUTURE** | Not available yet |

### Requirement: 25+ Tokens/Second

| Service | Likelihood | Notes |
|---------|-----------|-------|
| Spheron | **POSSIBLE** | With bare metal H100 cluster, optimized setup |
| Prime Intellect | **CHALLENGING** | Pipeline bubbles + internet latency |
| Exabits | **UNKNOWN** | Insufficient documentation |
| Bittensor | **UNLIKELY** | Network latency, consumer hardware |
| Flux | **NO** | Architecture not suited |
| Ritual | **FUTURE** | Not available yet |

### Requirement: TEE / Confidential Computing NOW

| Service | Available | Notes |
|---------|-----------|-------|
| **Spheron** | **YES** | H100/H200 with NVIDIA CC |
| **Bittensor Targon** | **YES** | SN4 confidential computing |
| Ritual | **PARTIAL** | Infernet supports TEE, but limited |
| Prime Intellect | **NO** | TOPLOC is verification only |
| Exabits | **NO** | Roadmap only |
| Flux | **NO** | Not planned |

### Requirement: Tensor Parallelism Across Nodes

| Service | Support | Notes |
|---------|---------|-------|
| Prime Intellect | **YES** | Pipeline parallelism native |
| Spheron | **USER-MANAGED** | Bare metal allows setup |
| Others | **NO** | Not designed for this |

---

## Recommendations

### Best Option for Your Requirements: **Spheron Network**

**Rationale:**
1. **Live TEE support** via NVIDIA H100/H200 Confidential Computing
2. **Sufficient VRAM** with 8x H100 bare metal (640GB)
3. **Competitive pricing** at $16.56/hr for 8x H100
4. **99% uptime** SLA
5. **Production-ready** with $12M+ ARR proving platform stability

**Limitations:**
- You must configure tensor parallelism yourself using standard frameworks
- Multi-node orchestration is not provided by the platform
- 25+ tok/sec will require optimization work

### Alternative: Centralized TEE Providers

For the specific combination of:
- 500GB+ model memory
- 25+ tok/sec throughput
- TEE requirements

Consider centralized providers with NVIDIA Confidential Computing:
- **Azure** (NC H100 v5 with Confidential VMs)
- **AWS** (coming with Vera Rubin in 2026)
- **CoreWeave** (H100 CC support)

These may provide better reliability and performance guarantees at potentially higher cost.

### Watch List

- **Ritual** - When mainnet launches, may offer best-in-class privacy
- **Exabits** - If platform matures, could be cost-effective option
- **Prime Intellect** - If they add TEE support, best for distributed inference

---

## Pricing Comparison (H100 80GB Per GPU-Hour)

| Provider | Type | Price |
|----------|------|-------|
| Spheron | Decentralized | $0.72+ (spot) |
| Vast.ai | Marketplace | $1.87 |
| RunPod | Community | $1.99 |
| Lambda | Centralized | $2.99 |
| AWS | Centralized | $3.90 |
| Azure | Centralized | $6.98 |

---

*Report compiled February 2026. Prices and capabilities subject to change. Verify current status before deployment decisions.*
