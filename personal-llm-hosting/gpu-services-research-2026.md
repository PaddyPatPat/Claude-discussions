# GPU Processing Services Research Report
## For Large MoE Model Inference with Privacy Requirements
**Research Date: February 2026**

---

## Executive Summary

This research evaluates GPU processing services for running large Mixture-of-Experts (MoE) LLM inference at 25+ tokens/second with strong privacy requirements (TEE acceptable). The target models require 230-600GB memory:

| Model | Total Params | Active Params | Memory (INT8) | Min GPUs |
|-------|-------------|---------------|---------------|----------|
| MiniMax M2.1 | 230B | ~10B | ~230GB | 4x H100 |
| GLM-4.7 | 355B | ~32B | ~355GB | 6x H100 |
| Kimi K2.5 | 1T | ~32B | ~600GB | 8x H100/H200 |

**Key Findings:**
1. **TEE for large models is limited** - Multi-GPU confidential computing clusters are still emerging
2. **Best TEE option NOW**: Phala Network (up to 8x H200 with NVIDIA CC) or Azure NCCadsH100v5
3. **Best price/performance**: io.net decentralized network (70% cheaper than AWS, TEE via Phala partnership)
4. **Mac Studio cluster remains competitive** for privacy (100% on-premise) but inflexible
5. **GB200 NVL72 arriving H1-H2 2026** will transform the economics (10x faster, rack-scale CC)

---

## Comparison Table: All Services

### Centralized Cloud GPU Services

| Service | Type | GPU Types | Max Cluster | TEE/Privacy | H100 $/hr | 500GB+ Models | Status |
|---------|------|-----------|-------------|-------------|-----------|---------------|--------|
| **Lambda Labs** | Central | H100, H200, B200 | 2,000+ | No (HIPAA yes) | $2.49-2.99 | Yes | Live |
| **CoreWeave** | Central | H100 SXM, H200 | 1,000+ | No | $4.25-6.15 | Yes | Live |
| **RunPod** | Central | H100, H200 | 32+ | No (SOC2 yes) | $1.99-2.69 | Yes | Live |
| **Vast.ai** | Central | H100, A100 | Variable | No (SOC2 yes) | $2.00-4.00 | Yes | Live |
| **Together AI** | Central | H100, H200, B200 | 100,000+ | Via VPC | Contact | Yes | Live |
| **Fireworks AI** | Central | H100, H200, MI300X | Contact | No (SOC2/HIPAA) | Contact | Yes | Live |
| **Modal** | Central | H100, A100 | Scales | No (SOC2/HIPAA) | ~$3.95 | Limited | Live |
| **Baseten** | Central | H100, B200 | 8+ multi-node | No (SOC2/HIPAA) | ~$6.50 | Yes | Live |
| **Anyscale** | Central | H100 | Scales (Ray) | Via Private EP | Contact | Yes | Live |
| **Crusoe** | Central | H100 SXM | 1,000+ | Unknown | Contact | Yes | Live |
| **Paperspace** | Central | H100 | 8x | No (SOC2) | $5.95 ($2.24 rsv) | Yes | Live |
| **FluidStack** | Central | H100, H200, B200 | 10,000+ | Unknown | $1.35+ | Yes | Live |
| **Azure** | Hyperscaler | H100 NVL | 1 (CC mode) | **YES - NOW** | ~$8.90 | No (single GPU) | Live |
| **Google Cloud** | Hyperscaler | H100 | 1 (CC mode) | **YES - NOW** | ~$3.00 | No (single GPU) | Live |

### Decentralized GPU Networks

| Service | Type | GPU Types | TEE/Privacy | Pricing vs AWS | 500GB+ Models | Status |
|---------|------|-----------|-------------|----------------|---------------|--------|
| **io.net** | Decentral | H100, H200, A100 | **YES (Phala)** | 70% cheaper | Yes | Live |
| **Akash Network** | Decentral | H100, A100 | Q1 2026 (AEP-65) | 60-75% cheaper | Yes (Starcluster H1 2026) | Live |
| **Spheron** | Decentral | H100, H200, B200 | **YES (NVIDIA CC)** | 47x cheaper claimed | Yes | Live |
| **Render** | Decentral | Mixed | No | Variable | No | Live |
| **Gensyn** | Decentral | Mixed | No | N/A | No (training focus) | Testnet |
| **Nosana** | Decentral | Consumer only | No | Cheap | No (24GB max) | Live |
| **Golem** | Decentral | Consumer | No | N/A | No | Beta |
| **Flux** | Decentral | Consumer + Premium | No | 90% cheaper | Limited | Live |
| **Prime Intellect** | Decentral | H100, RTX 4090 | No (verification yes) | N/A | Yes (distributed) | Preview |
| **Exabits** | Decentral | GB200, H200, H100 | Roadmap | Unknown | Possibly | Early |
| **Bittensor** | Decentral | Mixed | Targon subnet only | 85% cheaper | Limited | Live |
| **Ritual** | Decentral | N/A | **Core architecture** | N/A | Future | Devnet |

### TEE/Confidential Computing Specialists

| Service | TEE Technology | GPU Support | Attestation | Multi-GPU CC | Status |
|---------|---------------|-------------|-------------|--------------|--------|
| **Azure** | AMD SEV-SNP + NVIDIA CC | H100 NVL (94GB) | Yes (Azure + NRAS) | No (single GPU) | **GA** |
| **Google Cloud** | Intel TDX + NVIDIA CC | H100 (80GB) | Yes | No (single GPU) | **GA** |
| **Phala Network** | Intel TDX + NVIDIA CC | H100, H200, B200 | Yes | **Yes (1-8 GPUs)** | **Live** |
| **io.net + Phala** | Intel TDX + NVIDIA CC | H100, H200 | Yes | Yes | **Live** |
| **Spheron** | NVIDIA CC | H100, H200 | Yes | User-managed | **Live** |
| **Edgeless Systems** | AMD SEV-SNP/TDX + NVIDIA CC | Via Azure/GCP | Yes | Via cloud | Live |
| **Super Protocol** | Intel TDX + NVIDIA CC | H100, Blackwell | Yes | Yes | Live |

---

## Top 3 Recommendations for Your Use Case

### Recommendation 1: Phala Network (Best TEE Option NOW)
**For: Maximum privacy with multi-GPU support**

| Attribute | Details |
|-----------|---------|
| **Why** | Only provider with production multi-GPU TEE (up to 8x H200 = 1.1TB) |
| **TEE** | Intel TDX (CPU) + NVIDIA Confidential Computing (GPU) |
| **Attestation** | Yes - cryptographic proof for business contracts |
| **Pricing** | H100: $3.08/hr, H200: $3.50/hr |
| **8x H200 cluster** | ~$28/hr = ~$2,000/month at 72 hrs usage |
| **Performance** | 5-15% overhead vs bare metal |
| **Models supported** | 8x H200 (1.1TB) fits all three target models |

**Action:** Contact Phala to verify 8-GPU NVLink cluster availability in CC mode.

### Recommendation 2: io.net with Phala TEE Integration
**For: Best price with TEE**

| Attribute | Details |
|-----------|---------|
| **Why** | 70% cheaper than AWS with TEE via Phala partnership |
| **TEE** | Full-stack via Phala (Intel TDX + NVIDIA CC) |
| **Pricing** | 8x H100 cluster ~$3.35/hr vs AWS $98.32/hr |
| **Monthly cost** | ~$1,500-2,000 for 100 hrs (fits your budget) |
| **Scale** | 300,000+ GPUs in 130 countries |
| **Performance** | 99% of native in TEE mode (<1% penalty) |

**Action:** Deploy test workload on io.net to validate TEE attestation workflow.

### Recommendation 3: Mac Studio Cluster (Your Baseline)
**For: Maximum control, proven performance**

| Attribute | Details |
|-----------|---------|
| **Why** | 100% on-premise exceeds TEE requirements |
| **Cost** | $40,000 one-time + ~$100/month electricity |
| **Memory** | 4x 512GB = 1.5TB (fits all models) |
| **Performance** | Proven 25-32 tok/s on comparable models |
| **Privacy** | Complete data sovereignty |
| **Limitation** | No flexibility to scale, macOS-only |

**Verdict:** Mac Studio remains best for **guaranteed privacy** but io.net/Phala offers **flexibility + TEE** at lower ongoing cost.

---

## Best Hardware Contribution Options

### For Apple Silicon (M3 Ultra, M4 Max)
**Finding: NO viable earning platforms exist for Apple Silicon**

The GPU rental market is CUDA-dominated. Options:
- **Exo** (open-source): Self-host distributed inference, no earnings
- **Hyperlink**: Claims $4,300-34,700/year but has reliability/payment issues

**Recommendation:** Use Apple Silicon for self-hosting only, not for earning.

### For Consumer GPUs (RTX 4090, RTX 3090)

| Platform | Monthly Earnings | Setup | Credit Offset |
|----------|------------------|-------|---------------|
| **Vast.ai** | $500-1,500 | Ubuntu server | No (cash only) |
| **io.net** | Token-based | Docker | **Yes** |
| **Salad** | $30-180 | Windows app | No |
| **GamerHash** | $150-300 | Windows app | No |

**Best for credit offset:** io.net - IO tokens can directly pay for inference on the same platform.

### For Workstation GPUs (A6000, RTX 6000 Ada)

| Platform | Monthly Earnings | Requirements |
|----------|------------------|--------------|
| **Aethir** | $25,000-40,000/8-GPU node | KYC, staking, enterprise SLAs |
| **Akash** | ~$600/GPU (~$20/day) | Kubernetes setup |
| **RunPod Partner** | Varies | ECC RAM, 200Gbps networking |

---

## Privacy Leaders Available NOW

### Tier 1: Production-Ready Multi-GPU TEE

| Provider | Multi-GPU TEE | Memory Available | Attestation | Pricing |
|----------|---------------|------------------|-------------|---------|
| **Phala Network** | 1-8 GPUs | Up to 1.1TB (8x H200) | Yes | $3.08-3.50/GPU-hr |
| **io.net + Phala** | Yes | Cluster-based | Yes | 70% < AWS |
| **Spheron** | User-managed | 640GB (8x H100) | Yes | $0.72-16.56/hr |

### Tier 2: Single-GPU TEE (Insufficient for your models)

| Provider | GPU | Memory | Pricing |
|----------|-----|--------|---------|
| Azure NCCadsH100v5 | 1x H100 NVL | 94GB | $8.90/hr |
| Google Cloud A3 CC | 1x H100 | 80GB | $3.00/hr |

### Tier 3: Strong Certifications (No Hardware TEE)

| Provider | Certifications | Data Isolation |
|----------|---------------|----------------|
| Together AI VPC | SOC2, HIPAA | Deploy in your cloud |
| Baseten | SOC2 Type II, HIPAA, GDPR | Encryption, no TEE |
| Fireworks AI | SOC2 Type II, HIPAA | Zero retention option |
| CoreWeave | SOC2 Type II, ISO 27001, HIPAA | Physical isolation |

---

## Cost Comparison: Kimi K2.5 at 25+ tok/s for 100 hrs/month

### API Services (Managed)

| Provider | Speed | Cost/100 hrs | Notes |
|----------|-------|--------------|-------|
| Fireworks | 200 tok/s | ~$300-600 (token-based) | Fastest |
| Moonshot AI | 100 tok/s | ~$180-600 (token-based) | Official |
| Together AI | High | Contact | OpenAI-compatible |

**Token-based math:** At 25 tok/s for 100 hrs = 9M output tokens. At $3/M = $27 output + input costs.

### Self-Hosted GPU Clusters

| Configuration | Provider | Monthly (100 hrs) | TEE |
|---------------|----------|-------------------|-----|
| 8x H100 SXM | RunPod | ~$2,150 | No |
| 8x H100 SXM | Lambda | ~$1,915 | No |
| 8x H200 | Phala | ~$2,800 | **Yes** |
| 8x H100 | io.net | ~$335 | **Yes** |

### Mac Studio Cluster Break-Even Analysis

| Metric | Mac Studio | io.net (8x H100 TEE) |
|--------|------------|---------------------|
| Upfront | $40,000 | $0 |
| Monthly | ~$100 (electricity) | ~$335-2,000 |
| Break-even | N/A | 20-40 months |

**At $500/month cloud spend:** Mac Studio pays off in ~80 months (6.7 years)
**At $2,000/month cloud spend:** Mac Studio pays off in ~20 months (1.7 years)

**Verdict:** If you need >50 hours/month consistently for 2+ years, Mac Studio is cost-effective. For variable/bursty workloads, cloud is better.

---

## Emerging Options to Monitor (2026)

### Q1 2026

| Service | What's Coming | Impact |
|---------|---------------|--------|
| **Akash AEP-65** | Confidential Computing (TEE) | Major - decentralized TEE alternative |
| **Gensyn Mainnet** | Distributed training/inference | Training option |

### H1 2026

| Service | What's Coming | Impact |
|---------|---------------|--------|
| **Akash Starcluster** | GB200 NVL72 (7,200 GPUs, 130 TB/s) | Transformative for trillion-param models |
| **AWS P6e-GB200** | Ultra Servers | 30x faster for trillion-param |
| **NVIDIA GB200 NVL72** | General availability | 10x faster, 1/10th cost per token |

### H2 2026

| Service | What's Coming | Impact |
|---------|---------------|--------|
| **NVIDIA Rubin** | Rack-scale Confidential Computing | Solves multi-GPU TEE problem |
| **CoreWeave Rubin** | First Rubin adopter | Enterprise TEE at scale |
| **Ritual Mainnet** | Best-in-class TEE architecture | Privacy-first inference |

---

## Risk Assessment

### Recommendation 1: Phala Network

| Risk Type | Level | Details |
|-----------|-------|---------|
| **Technical** | Medium | Verify 8-GPU NVLink in CC mode works for your models |
| **Availability** | Low | Production service, 1.34B+ tokens processed |
| **Business** | Medium | Smaller company, verify SLAs and support |

### Recommendation 2: io.net + Phala

| Risk Type | Level | Details |
|-----------|-------|---------|
| **Technical** | Medium | Decentralized network, variable node quality |
| **Availability** | Low-Medium | 300K+ GPUs but attestation via partner |
| **Business** | Medium | Token-based economics, IO token volatility |

### Recommendation 3: Mac Studio Cluster

| Risk Type | Level | Details |
|-----------|-------|---------|
| **Technical** | Low | Proven benchmarks, mature ecosystem (Exo 1.0) |
| **Availability** | Low | You own it, 100% control |
| **Business** | Low | Apple is stable, resale value exists |
| **Flexibility** | High | Cannot scale up, locked to 1.5TB |

### Wait-and-See: Akash TEE (March 2026)

| Risk Type | Level | Details |
|-----------|-------|---------|
| **Technical** | Medium | AEP-65 is well-designed on paper |
| **Availability** | High | Software launches often slip |
| **Business** | Low | Well-funded, active development |

---

## Final Recommendations

### If Privacy is Non-Negotiable (TEE Required)

1. **Now:** Deploy on **Phala Network** or **io.net with Phala TEE**
2. **Validate:** Test attestation workflow, verify business contract acceptance
3. **Fallback:** Mac Studio cluster for 100% on-premise

### If Strong Certifications Suffice (SOC2/HIPAA)

1. **API-first:** Use **Fireworks** (fastest) or **Together AI** (most flexible)
2. **Self-host:** **RunPod** or **Lambda** for cost optimization
3. **Enterprise:** **CoreWeave** or **Baseten** for SLAs

### If Cost is Primary Concern

1. **Decentralized:** **io.net** (70% cheaper than AWS)
2. **Spot/auction:** **Vast.ai** marketplace
3. **Long-term:** Mac Studio cluster (break-even at 20-40 months)

### Strategic Wait

If your deadline allows flexibility:
- **Wait for Akash AEP-65** (March 2026) - decentralized TEE
- **Wait for GB200** (H1 2026) - 10x performance improvement
- **Wait for Rubin** (H2 2026) - rack-scale confidential computing

---

## Sources

### Centralized Providers
- [Lambda Labs Pricing](https://lambda.ai/pricing)
- [CoreWeave Pricing](https://www.coreweave.com/pricing)
- [RunPod Pricing](https://www.runpod.io/pricing)
- [Together AI](https://www.together.ai/pricing)
- [Fireworks AI](https://fireworks.ai/pricing)
- [Baseten Pricing](https://www.baseten.co/pricing/)

### Decentralized Networks
- [Akash Network](https://akash.network/)
- [io.net](https://io.net/)
- [Phala Network](https://phala.com/gpu-tee)
- [Spheron Network](https://www.spheron.network/)

### TEE/Confidential Computing
- [Azure Confidential GPU](https://learn.microsoft.com/en-us/azure/confidential-computing/gpu-options)
- [Google Cloud Confidential VMs](https://docs.cloud.google.com/confidential-computing/confidential-vm/docs/confidential-vm-overview)
- [NVIDIA Confidential Computing](https://www.nvidia.com/en-us/data-center/solutions/confidential-computing/)
- [NVIDIA NVTrust](https://github.com/NVIDIA/nvtrust)

### Large Model Inference
- [Kimi K2 on OpenRouter](https://openrouter.ai/moonshotai/kimi-k2-thinking)
- [Cerebras GLM-4.7](https://www.cerebras.ai/blog/glm-4-7)
- [vLLM Parallelism](https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html)

### Hardware Contribution
- [Vast.ai Hosting](https://vast.ai/hosting)
- [io.net Provider](https://io.net/)
- [Salad](https://salad.com/)
