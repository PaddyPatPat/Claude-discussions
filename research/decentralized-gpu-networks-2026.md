# Decentralized GPU Networks for LLM Inference - Research Report
**Date:** February 2026
**Use Case:** Large MoE models (230B-1T parameters, 230-600GB memory), 25+ tokens/sec, TEE/privacy required
**Budget:** $500-2,000/month or $40K one-time

---

## Executive Summary

| Network | Status | Best GPU Available | Pricing vs. Centralized | TEE Available Now? | Large Model (500GB+) | Tensor Parallelism |
|---------|--------|-------------------|------------------------|-------------------|---------------------|-------------------|
| **Akash Network** | Live (Production) | H100 SXM5, GB200 (2026) | 60-75% cheaper | Q1 2026 (AEP-65) | Via Starcluster | Limited to single node |
| **Render Network** | Live (Production) | H200, MI300X | 50-70% cheaper | Not available | Limited | No |
| **io.net** | Live (Production) | H100, H200, A100 | 70% cheaper | Yes (via Phala) | Cluster support | Yes |
| **Gensyn** | Testnet (Mainnet Q1 2026) | Consumer + Datacenter | TBD | No | Training focus | Distributed training |
| **Nosana** | Live (Production) | RTX 4090/3090 (consumer) | 2.5x cheaper | No | No | No |
| **Golem Network** | Beta | 30xx+ consumer GPUs | $0.10-0.50/hr | No | No | In development |

---

## 1. Akash Network

### Operational Status
**LIVE - Production Grade**

AkashML launched November 2025 as a fully managed AI inference service. The platform provides:
- OpenAI-compatible API
- Automated scaling across ~65 datacenters
- 99% uptime claimed
- Serverless experience abstracting Kubernetes complexity

### GPU Types Available
| GPU | Availability | Pricing |
|-----|--------------|---------|
| H100 SXM5 | Live | $1.18-$2.53/hr |
| H100 PCIe | Live | ~$1.50/hr |
| A100 80GB | Live (91% utilized) | ~$0.80-1.20/hr |
| RTX 4090 clusters | Live | Significantly cheaper |
| **GB200 (Starcluster)** | Coming H1 2026 | TBD |

**Starcluster Initiative:** 7,200 GB200 GPUs funded via $75M Starbonds offering, operated by enterprise-grade "Nodekeepers." Hardware coming online late 2025 into early 2026.

### Pricing Comparison
- **60-75% cheaper than AWS/GCP**
- H100: $1.18-2.53/hr vs. AWS ~$4-8/hr
- Clustered RTX 4090s: Up to 75% cost reduction vs. H100s for batch inference

### TEE / Confidential Computing
**AEP-65 Status: Target Q1 2026**

Technical implementation planned:
- Intel TDX/SGX and AMD SEV-SNP for CPU TEE
- NVIDIA NVTRUST SDK integration for H100/H200 GPUs in CC-on mode
- Hardware attestation via AEP-29 for H100/H200 GPUs
- Full system coverage: CPU, GPU, and memory encryption

**Current Status:** Not yet available. March 2026 target per roadmap.

### Large Model Support (500GB+)
**LIMITED - Single Node Constraint**

- Current deployments support multi-GPU within single nodes
- No documented tensor parallelism across multiple provider nodes
- vLLM deployments possible but constrained to single-node TP
- **Starcluster with GB200 NVL72** (72-GPU NVLink domain) could handle trillion-parameter models when live

**For 500GB+ models:** Would require Starcluster GB200 infrastructure (H1 2026)

### Tensor Parallelism
- Single-node multi-GPU: Supported via vLLM
- Multi-node tensor parallelism: Not documented for decentralized providers
- Pipeline parallelism across nodes: Not available

### Data Privacy Architecture
- Provider-side workload isolation
- Container-based deployment sandboxing
- TEE protection (when AEP-65 ships) will include:
  - Processor-level isolation
  - Encrypted memory
  - Attestation proofs

### Network Reliability
- 99% uptime claimed for AkashML
- No formal SLA with financial penalties
- Provider stake slashing for poor performance
- 57% peak utilization on H100s (Q2 2025)

### Assessment for Your Use Case
| Requirement | Score | Notes |
|-------------|-------|-------|
| Large MoE (500GB+) | ⚠️ Limited | Need Starcluster GB200 |
| 25+ tok/sec | ✅ Possible | With H100 clusters |
| TEE/Privacy | ⚠️ March 2026 | AEP-65 roadmap |
| Budget fit | ✅ Yes | Well within $500-2000/mo |

---

## 2. Render Network

### Operational Status
**LIVE - Production (Rendering Focus, AI Expanding)**

- Historically focused on 3D rendering (65M+ frames rendered)
- December 2025: Launched Dispersed.com for AI/ML workloads
- 5,600 node operators with 85-95% utilization
- 1.5M frames/month processing

### GPU Types Available
| GPU | Availability | Notes |
|-----|--------------|-------|
| Consumer GPUs | Widely available | Primary node type |
| NVIDIA H200 | Enterprise tier | Via RNP-021 |
| AMD MI300X | Enterprise tier | Via RNP-021 |

**Enterprise GPU onboarding** (H200, MI300X) occurred via RNP-021 proposal.

### Pricing Comparison
- **50-70% cheaper than AWS/GCP**
- Burn Mint Equilibrium (BME) pricing model
- Pay only for actual GPU time, billed per minute
- No surprise fees or opaque pricing tiers
- Dynamic pricing under RNP-018 for AI compute layer

### TEE / Confidential Computing
**NOT AVAILABLE**

- No TEE implementation documented
- No confidential computing roadmap found
- Focus remains on rendering quality verification rather than data privacy

### Large Model Support (500GB+)
**LIMITED**

Key limitations identified:
- Current infrastructure optimized for rendering, not LLM inference
- "Startups not fully equipped to handle deployment of resource-intensive inference across distributed retail GPUs"
- Latency limitations for real-time workloads
- Data transfer constraints across distributed nodes

### Tensor Parallelism
**NOT SUPPORTED**

- No multi-node tensor parallelism documented
- Nodes operate independently for rendering jobs
- AI compute subnet (RNP-019) is nascent

### Data Privacy Architecture
- No TEE or confidential computing
- Standard container isolation
- Proof-of-compute verification (rendering quality, not privacy)

### Network Reliability
- 85-95% utilization rates (high demand)
- No formal SLA documented
- Competition from centralized providers remains a risk
- Enterprise GPU nodes may improve reliability

### Assessment for Your Use Case
| Requirement | Score | Notes |
|-------------|-------|-------|
| Large MoE (500GB+) | ❌ Poor | Not designed for this |
| 25+ tok/sec | ⚠️ Uncertain | Latency limitations |
| TEE/Privacy | ❌ No | Not available |
| Budget fit | ✅ Affordable | But capabilities lacking |

**NOT RECOMMENDED** for your use case.

---

## 3. io.net

### Operational Status
**LIVE - Production**

- 300,000+ verified GPUs globally
- Cluster deployment in under 2 minutes
- Available in 130+ countries
- Workloads: batch inference, model serving, parallel training, RL

### GPU Types Available
| GPU | Availability | Notes |
|-----|--------------|-------|
| H100 | Available | Enterprise tier |
| H200 | Available | TEE-capable |
| A100 | Widely available | Standard tier |
| RTX 4090 | Abundant | Consumer tier |
| RTX 3090 | Abundant | Consumer tier |

**HyperGPU feature:** Integrates consumer RTX 4090s with enterprise hardware for hybrid clusters.

### Pricing Comparison
| Configuration | io.net | AWS | Savings |
|---------------|--------|-----|---------|
| 8-GPU H100 cluster | $3.35/hr | $98.32/hr | 97% |
| Consumer 4090 inference | 75% cheaper than H100 | - | - |

**Up to 70% savings vs. AWS/GCP** claimed across the platform.

### TEE / Confidential Computing
**AVAILABLE NOW (via Phala Network Partnership)**

Strategic partnership with Phala Network provides:
- Full-stack TEE protection:
  - Intel TDX for CPU/memory protection
  - NVIDIA Confidential Computing for GPU encryption
- H100/H200 TEE mode: 99% efficiency vs. native performance
- Model weights, training data, inference results encrypted during computation

**Benchmark Results:**
- TEE mode performance penalty: <1% on H100/H200
- Validated for real-world, high-performance AI applications
- LLaMA 3 and Microsoft Phi deployed successfully in TEE

### Large Model Support (500GB+)
**SUPPORTED - Cluster Architecture**

- GPU cluster support designed for large-scale ML
- Parallel training and inference workloads
- Integration with decentralized GPU marketplace
- Ray cluster deployment supported

**Key capability:** Dynamic scaling with no contracts or lock-in

### Tensor Parallelism
**SUPPORTED**

- GPU cluster architecture supports tensor parallelism
- Low-latency interconnects within clusters
- Proprietary clustering technology for latency minimization
- Bare metal, container, and Ray cluster options

### Data Privacy Architecture
- Phala TEE integration provides hardware-backed privacy
- Encrypted memory during computation
- Attestation proofs for compute verification
- Zero-knowledge of data by provider nodes in TEE mode

### Network Reliability
- No formal SLA with financial penalties documented
- Staking-based provider incentive alignment
- Under-performing providers can be slashed
- IDE overhaul (Q2 2026) will link emissions to actual compute demand

### Assessment for Your Use Case
| Requirement | Score | Notes |
|-------------|-------|-------|
| Large MoE (500GB+) | ✅ Yes | GPU cluster support |
| 25+ tok/sec | ✅ Yes | H100/H200 clusters |
| TEE/Privacy | ✅ Yes | Phala partnership live |
| Budget fit | ✅ Yes | 70% cheaper than cloud |

**STRONG CANDIDATE** - Best current combination of capabilities for your requirements.

---

## 4. Gensyn

### Operational Status
**TESTNET - Mainnet Expected Q1 2026**

- Public testnet launched March 2025
- 150,000+ users
- ~40,000 active nodes
- 800,000+ models trained
- $67M funding (a16z led $43M Series A)
- $16.14M token sale December 2025 (7,412 participants)

**Mainnet timing:** Co-founder stated "3-4 weeks away" as of November 2025.

### GPU Types Available
| GPU Type | Status | Notes |
|----------|--------|-------|
| Consumer GPUs | Live on testnet | RTX series, gaming PCs |
| Apple Silicon | Live on testnet | Mac devices |
| Datacenter GPUs | Live on testnet | H100, A100 |
| Custom ASICs | Supported | Mining hardware |
| SoC devices | Supported | Mobile/tablet |

**Unique:** Only network designed to span from smartphones to datacenters.

### Pricing Comparison
**TBD - Pre-Mainnet**

- No production pricing available
- Testnet uses token incentives, not real pricing
- Likely competitive when live based on compute aggregation model

### TEE / Confidential Computing
**NOT AVAILABLE**

- No TEE implementation documented
- Privacy model focuses on trustless verification rather than confidential computing
- Verde (verification) and proof systems instead of encryption

### Large Model Support (500GB+)
**TRAINING FOCUS - Limited Inference**

Key architecture insights:
- Designed primarily for training, not inference
- Distributed training across heterogeneous devices
- RL Swarm for collaborative reinforcement learning
- "Boundaries between inference, training, computing, and data become blurred"

**For inference:** Other platforms better suited; Gensyn optimized for training

### Tensor Parallelism
**DISTRIBUTED TRAINING ONLY**

- Megatron-style parallelism for training
- NoLoCo and SkipPipe for communication efficiency
- Not optimized for inference parallelism
- Verification focuses on training proofs

### Data Privacy Architecture
- Trustless verification (Verde)
- Proof-of-work style attestation
- No encryption-in-use (not TEE-based)
- Smart contract coordination for task allocation

### Network Reliability
- Testnet only - no production SLA
- Decentralized coordination may introduce latency
- Mainnet auditing pending
- Economic security model still being validated

### Assessment for Your Use Case
| Requirement | Score | Notes |
|-------------|-------|-------|
| Large MoE (500GB+) | ⚠️ Training only | Not inference-optimized |
| 25+ tok/sec | ❌ No | Training focus |
| TEE/Privacy | ❌ No | Trustless, not confidential |
| Budget fit | ❓ TBD | Not yet priced |

**NOT RECOMMENDED** for inference use case. Consider for training workloads post-mainnet.

---

## 5. Nosana

### Operational Status
**LIVE - Production on Solana**

- Decentralized GPU grid on Solana blockchain
- ~2,000 active nodes
- Dynamic pricing marketplace
- Payments streamed per second in $NOS tokens

### GPU Types Available
| GPU | Availability | Notes |
|-----|--------------|-------|
| RTX 4090 | Most common | Consumer flagship |
| RTX 3090 | Very common | Second most popular |
| Other RTX | Available | 20xx, 30xx series |
| **Enterprise GPUs** | Limited | Not primary focus |

**Key insight:** Nosana explicitly leverages underutilized consumer-grade hardware, NOT datacenter GPUs.

### Pricing Comparison
- **2.5x cheaper than A100** for equivalent inference throughput
- Dynamic marketplace pricing
- Provider-set rates with automatic optimization
- Up to 6x cost savings claimed vs. cloud

**Consumer GPU Economics:**
- RTX 4090 inference at 2.5x lower cost than A100
- Same ROI per inference operation
- Suitable for smaller models, not 500GB+ workloads

### TEE / Confidential Computing
**NOT AVAILABLE**

- No TEE implementation
- No confidential computing roadmap documented
- Focus on cost efficiency, not privacy guarantees

### Large Model Support (500GB+)
**NOT SUPPORTED**

Critical limitations:
- RTX 4090: 24GB VRAM
- RTX 3090: 24GB VRAM
- No multi-GPU tensor parallelism documented
- Maximum practical model size: ~40B parameters (with quantization)

**Your 230-600GB models cannot run on Nosana infrastructure.**

### Tensor Parallelism
**NOT SUPPORTED**

- Single GPU deployments only
- No multi-node coordination for inference
- Job matching is 1:1 (one job, one GPU)

### Data Privacy Architecture
- Standard container isolation
- No encryption-in-use
- Solana-based payment verification
- Provider reputation system

### Network Reliability
- Decentralized - no central SLA
- Consumer hardware may have variable availability
- 2026 roadmap focuses on "reducing friction" and scaling

### Assessment for Your Use Case
| Requirement | Score | Notes |
|-------------|-------|-------|
| Large MoE (500GB+) | ❌ No | Max 24GB VRAM per node |
| 25+ tok/sec | ❌ No | Consumer GPU limits |
| TEE/Privacy | ❌ No | Not available |
| Budget fit | ✅ Very cheap | But unusable for your models |

**NOT SUITABLE** - Hardware incapable of running your models.

---

## 6. Golem Network

### Operational Status
**BETA - GPU Provider in Testing**

- First phase of GPU beta testing completed (~50 testers)
- CPU compute network established since 2016
- GPU provider requires NVIDIA 30xx+ with 8GB+ VRAM
- **Modelserve AI development temporarily suspended**

### GPU Types Available
| GPU | Availability | Notes |
|-----|--------------|-------|
| RTX 30xx series | Beta testing | 8GB+ VRAM required |
| RTX 40xx series | Beta testing | Consumer tier |
| Enterprise H100 | Roadmap | Not widely available yet |

**Hardware Requirements:**
- IOMMU support required
- NVIDIA 30xx+ (8GB+ VRAM minimum)
- RAM at least equal to GPU VRAM (2x recommended)

### Pricing Comparison
| GPU Tier | Estimated Pricing |
|----------|-------------------|
| Mid-tier (consumer) | ~$0.10/hr |
| Enterprise H100 | ~$0.50/hr |

**Note:** Pricing is estimated; actual marketplace rates will vary.

### TEE / Confidential Computing
**NOT AVAILABLE**

- No TEE implementation
- No confidential computing roadmap
- Focus on general GPU compute accessibility

### Large Model Support (500GB+)
**NOT SUPPORTED**

Limitations:
- Consumer GPU focus (24GB max per GPU)
- Multi-GPU support "being developed"
- No multi-node tensor parallelism
- Golem-Workers provides raw GPU access but no orchestration

**Modelserve validation:** Testing showed consumer GPUs suitable for "Whisper, Stable Diffusion, and small LLMs" - NOT large models.

### Tensor Parallelism
**IN DEVELOPMENT**

- Multi-GPU support planned but not live
- Provider management panel development ongoing
- Currently single-GPU deployments only

### Data Privacy Architecture
- Standard container isolation
- No TEE or encryption-in-use
- Task-level sandboxing

### Network Reliability
- Beta phase - no SLA
- GamerHash partnership for AI Provider (Windows only)
- Network maturity for GPU is low

### Assessment for Your Use Case
| Requirement | Score | Notes |
|-------------|-------|-------|
| Large MoE (500GB+) | ❌ No | Consumer GPU only |
| 25+ tok/sec | ❌ No | Hardware inadequate |
| TEE/Privacy | ❌ No | Not available |
| Budget fit | ✅ Very cheap | But not functional |

**NOT SUITABLE** - Still in beta, inadequate for your requirements.

---

## Comparative Analysis

### For Your Specific Requirements

**Requirements Recap:**
- 230B-1T parameter MoE models
- 230-600GB memory footprint
- 25+ tokens/second throughput
- TEE/confidential computing required
- Budget: $500-2,000/month or $40K one-time

### Capability Matrix

| Network | Large Models | Speed | TEE | Budget | Overall |
|---------|--------------|-------|-----|--------|---------|
| io.net | ✅ | ✅ | ✅ | ✅ | **Best Choice** |
| Akash Network | ⚠️ Q1 2026 | ✅ | ⚠️ Q1 2026 | ✅ | Second Choice |
| Gensyn | Training only | ❌ | ❌ | TBD | Not for inference |
| Render Network | ❌ | ⚠️ | ❌ | ✅ | Not suitable |
| Nosana | ❌ | ❌ | ❌ | ✅ | Not suitable |
| Golem Network | ❌ | ❌ | ❌ | ✅ | Not suitable |

### Top Recommendations

#### 1. io.net (RECOMMENDED)
**Why:**
- Only platform with live TEE via Phala Network partnership
- H100/H200 clusters available now
- 99% performance in TEE mode
- 70% cheaper than centralized alternatives
- GPU cluster architecture supports 500GB+ models

**Estimated Monthly Cost (500GB model at 25 tok/sec):**
- 8x H100 cluster: ~$1,500-2,000/month at 70% utilization
- Well within your budget

#### 2. Akash Network (WAIT FOR Q1 2026)
**Why:**
- Strong roadmap with AEP-65 confidential computing
- Starcluster GB200 (72-GPU NVLink) perfect for trillion-parameter models
- H100 SXM5 currently available at $1.18-2.53/hr
- AkashML provides managed experience

**When Ready:**
- TEE: March 2026 target
- GB200 Starcluster: Late Q1 2026

**Estimated Monthly Cost:**
- Current H100: ~$1,000-1,500/month
- With Starcluster GB200: TBD but likely within budget

---

## Budget Analysis

### Monthly Rental Approach ($500-2,000/month)

| Option | Configuration | Est. Cost/Month | Notes |
|--------|--------------|-----------------|-------|
| io.net H100 cluster | 8x H100 SXM | $1,500-2,000 | Best for immediate needs |
| Akash H100 | 4x H100 SXM5 | $1,000-1,500 | Need to wait for TEE |
| Akash Starcluster | GB200 access | TBD | When available |

### One-Time Purchase ($40K)
At $40K, you cannot purchase sufficient hardware to run 500GB+ models locally:
- Used H100: ~$27-40K per GPU
- Need minimum 8x H100 (640GB aggregate) = $216K-$320K
- **Rental is the only viable path for your budget**

---

## Privacy/Security Deep Dive

### TEE Comparison

| Platform | CPU TEE | GPU TEE | Full-Stack | Performance Impact |
|----------|---------|---------|------------|-------------------|
| io.net + Phala | Intel TDX | NVIDIA CC (H100/H200) | ✅ Yes | <1% |
| Akash (Q1 2026) | Intel TDX/SGX, AMD SEV | NVIDIA NVTRUST | ✅ Planned | TBD |
| Others | ❌ | ❌ | ❌ | N/A |

### What TEE Provides
1. **Memory Encryption:** All data encrypted while in GPU/CPU memory
2. **Attestation:** Cryptographic proof of hardware integrity
3. **Isolation:** Workloads isolated from provider access
4. **Model Protection:** Weights never exposed in plaintext

### Privacy Risks Without TEE
- Provider can inspect memory contents
- Model weights could be extracted
- Input/output data visible to provider
- No cryptographic guarantees

---

## Action Plan

### Immediate (Now)
1. **Start with io.net + Phala TEE** for production workloads requiring privacy
2. Deploy on H100/H200 clusters in TEE mode
3. Validate 25+ tok/sec throughput for your specific models

### Short-Term (Q1 2026)
1. Monitor Akash AEP-65 confidential computing launch
2. Evaluate Akash Starcluster GB200 availability
3. Consider hybrid approach: io.net for privacy-critical, Akash for cost optimization

### Long-Term
1. Gensyn mainnet for training workloads (when needed)
2. Watch for Golem GPU maturation
3. Re-evaluate as decentralized networks improve multi-node parallelism

---

## Sources

### Akash Network
- [Akash AEP-65 Confidential Computing Roadmap](https://akash.network/roadmap/aep-65/)
- [AkashML Launch](https://akash.network/blog/akashml-managed-ai-inference-on-the-decentralized-supercloud/)
- [State of Akash Q3 2025 - Messari](https://messari.io/report/state-of-akash-q3-2025)
- [Akash GPU Pricing - GPU Compare](https://gpucompare.com/providers/akash-network)

### Render Network
- [Understanding Render Network - Messari](https://messari.io/report/understanding-the-render-network-a-comprehensive-overview)
- [Render Network Pricing](https://rendernetwork.com/pricing)
- [Render Network Strategic AI Pivot](https://www.ainvest.com/news/render-network-strategic-ai-compute-pivot-path-8-2601/)

### io.net
- [io.net GPU Cluster Guide](https://io.net/blog/what-is-a-gpu-cluster)
- [io.net Pricing Model](https://developers.io.net/docs/pricing-model)
- [io.net and Phala TEE Partnership](https://phala.com/posts/phala-and-ionet-create-strategic-partnership-to-enhance-gputee-)
- [io.net Overview - Messari](https://messari.io/report/understanding-io-net-a-comprehensive-overview)

### Gensyn
- [Gensyn Documentation](https://docs.gensyn.ai/)
- [Gensyn Testnet Overview](https://docs.gensyn.ai/testnet)
- [Gensyn Architecture Breakdown](https://university.mitosis.org/breaking-down-gensyn-core-features-and-architecture/)
- [Gensyn Mainnet Q1 2026](https://egw.news/crypto/news/31585/most-anticipated-mainnet-launches-in-q1-2026-kAeXi2akn)

### Nosana
- [Nosana Official Site](https://nosana.com/)
- [Nosana GPU Marketplace](https://explore.nosana.com/markets/)
- [Nosana LLM Benchmarking](https://nosana.com/blog/llm_benchmarking_cost_efficient_performance/)
- [Nosana 2025 Wrapped](https://nosana.com/blog/wrapped_2025/)

### Golem Network
- [Golem AI/GPU Roadmap](https://blog.golem.network/ai-gpu-roadmap-update/)
- [Golem GPU Provider Docs](https://docs.golem.network/docs/providers/gpu/gpu-golem-live)
- [Golem AI Page](https://golem.network/ai)

### Phala Network (io.net TEE Partner)
- [Phala GPU TEE](https://phala.com/gpu-tee)
- [Phala H100 TEE Benchmarks](https://phala.com/posts/confidential-computing-on-nvidia-h100-gpu-a-performance-benchmark-study)
- [Host LLM in TEE - Phala Docs](https://docs.phala.network/confidential-ai-inference/host-llm-in-tee)

### General GPU Pricing
- [H100 Price Guide 2026 - JarvisLabs](https://docs.jarvislabs.ai/blog/h100-price)
- [GPU Price Comparison 2026](https://getdeploying.com/gpus)
- [Top Cloud GPU Providers 2026 - RunPod](https://www.runpod.io/articles/guides/top-cloud-gpu-providers)
