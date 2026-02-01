# GPU & Apple Silicon Contribution Programs Research (2026)

This document details services that allow contributing personal hardware (GPUs, Apple Silicon) to earn credits, cryptocurrency, or cash.

---

## Executive Summary

| Platform | Hardware Support | Est. Monthly Earnings | Setup Complexity | Credit Offset | Payout Method |
|----------|------------------|----------------------|------------------|---------------|---------------|
| **Vast.ai** | NVIDIA (GTX 10xx+), AMD | $500-1,500/GPU | Medium (Ubuntu) | No | PayPal, Wise ($20 min) |
| **Salad** | NVIDIA (GTX 10xx+), AMD (R9 300+) | $30-180/GPU | Low (Windows app) | No | PayPal, Gift Cards ($5 min) |
| **io.net** | NVIDIA (RTX 3090+), H100, A100 | Varies (token-based) | Medium (Docker) | Yes (IO tokens) | IO Token (200 IO stake req) |
| **Aethir** | Enterprise GPUs (H100, H200) | $25,000-40,000/8-GPU node | High (KYC, staking) | Yes (ATH tokens) | ATH Token |
| **Akash** | Enterprise GPUs (A100, H100) | ~$20/GPU/day avg | High (K8s) | Yes (AKT tokens) | AKT Token |
| **GamerHash** | NVIDIA GPUs | Up to $150-300/month | Low (Windows app) | No | GUSD (crypto) |
| **Hyperlink** | CPU & GPU (cross-platform) | $4,300-34,700/year claimed | Low (desktop app) | No | PayPal, Bank Transfer |
| **Q Blocks** | NVIDIA GPUs | Varies | Medium | No | Contact for details |
| **Fluence** | Enterprise (Tier 3-4 DCs) | Revenue share | High (DC required) | Yes (FLT tokens) | FLT Token |
| **Exo** | Apple Silicon, NVIDIA, etc. | N/A (self-hosting) | Medium (open source) | N/A | N/A (cost savings only) |

---

## Detailed Platform Analysis

---

### 1. Vast.ai

**Website:** https://vast.ai/hosting

**Overview:**
Vast.ai operates like "Airbnb for GPUs" - a peer-to-peer marketplace where individuals rent out their hardware through a competitive marketplace. You set your own prices and schedules.

**Supported Hardware:**
- NVIDIA GPUs: GTX 10-series and newer, RTX 20/30/40 series, Titan series
- AMD GPUs: Supported (specific models vary)
- Professional: A100, H100, RTX A6000, RTX 6000 Ada
- **Apple Silicon: NOT SUPPORTED** (CUDA-dependent platform)

**Minimum Requirements:**
- Operating System: Ubuntu 22.04 Server
- Storage: 100GB ext4 root partition + XFS data partition
- SSH Server required
- For Certified Data Center tier: ISO 27001 certification minimum

**Earnings Potential:**
- RTX 3090/4090: $500-1,500/month at 40-60% utilization
- Dynamic pricing: typically $0.80-$2.00/hour for consumer GPUs
- RTX 4090 market rate: $0.17-$0.32/hour on platform
- A6000/professional cards: Higher rates available
- **No hosting fees** (removed as of June 2024)

**Setup Complexity:** Medium
- Requires Ubuntu server installation
- Custom disk partitioning
- SSH configuration
- Docker containerization

**Payout Details:**
- Methods: PayPal, Wise (international)
- Minimum payout: $20
- Billing cycle: Weekly (every Friday)
- No hosting fee deduction

**Credit Offset:** No - earnings are cash-based, not platform credits

**Network Size:** Thousands of providers globally

**Sources:**
- [Vast.ai Hosting](https://vast.ai/hosting)
- [Vast.ai Documentation](https://docs.vast.ai/documentation/host/hosting-overview)
- [Fluence GPU Marketplace Comparison](https://www.fluence.network/blog/best-gpu-rental-marketplaces/)

---

### 2. SaladCloud

**Website:** https://salad.com/

**Overview:**
Salad is a worldwide compute-sharing community of 450,000+ "Salad Chefs" who share resources from their idle PCs. The platform runs containerized workloads via WSL2.

**Supported Hardware:**
- NVIDIA: GTX 1000-series and greater
- AMD: R9 300-series and greater (e.g., RX 470, RX 6700 XT)
- **Apple Silicon: NOT SUPPORTED** (Windows/WSL2 required)

**Minimum Requirements:**
- Windows OS with WSL2 support
- GPU with sufficient VRAM for container workloads
- Stable internet connection
- Specific GPU models must meet core power thresholds

**Earnings Potential:**
- RTX 4090/3090 Ti/3090: Up to $180/month (guaranteed best workloads)
- Average users: $30-200+/month
- Platform takes ~25% cut (reflected in earnings)
- Actual rates depend on market demand and availability

**Setup Complexity:** Low
- Download Windows desktop app
- One-click start/stop
- Uses Windows Subsystem for Linux 2 (WSL2)
- Container Engine (SCE) handles workload deployment

**Payout Details:**
- Methods: PayPal, Visa/Mastercard digital cards, Steam gift cards, Discord Nitro, various gift cards
- Minimum threshold: ~$5 for most options
- PayPal processing: up to 24 hours

**Credit Offset:** No - earns "Salad Balance" redeemable for rewards/PayPal

**Network Size:** 450,000+ active GPU providers, 60,000+ daily active GPUs

**Sources:**
- [Salad Official](https://salad.com/)
- [How Much Can I Earn with Salad](https://support.salad.com/article/60-how-much-can-i-earn-with-salad)
- [Salad PayPal Redemption](https://support.salad.com/article/612-how-to-redeem-paypal)

---

### 3. io.net

**Website:** https://io.net/

**Overview:**
io.net is a decentralized GPU cloud network integrating idle GPU resources worldwide. The platform connects global customers with suppliers through containerized virtual networks. Focused on AI/ML workloads.

**Supported Hardware:**
- NVIDIA: RTX 4090, RTX 3090, H100, H200, A100, and data center GPUs
- AMD: Ryzen series mentioned
- **Apple Silicon: Status unclear** (documentation needed)

**Minimum Requirements:**
- Minimum uptime: 5 hours to maintain block reward eligibility
- Base stake: 200 IO tokens per chip
- Stake adjusted by GPU/CPU model earning multiplier
- Connectivity: Higher bandwidth = better rewards

**Earnings Potential:**
- Token-based rewards via Proof of Time-Lock (PoTL)
- RTX 4060 server: ~$1/hour reported
- Factors affecting earnings: bandwidth, uptime, device multiplier, job hours, GPU model
- New tokenomics (IDE) launching Q2 2026: demand-driven model tied to compute usage

**Setup Complexity:** Medium
- Docker container deployment
- IO Worker software installation
- Staking requirement (200 IO per chip minimum)
- Real-time monitoring via platform

**Payout Details:**
- Method: IO Token (cryptocurrency)
- Penalties for uptime failures and fraudulent activity
- Automated payouts
- Token can be exchanged on crypto markets

**Credit Offset:** Yes - IO tokens earned can be used to pay for compute on the platform

**Network Size:** 130+ countries, claims to be largest decentralized GPU network

**2026 Roadmap:**
- Q2 2026: Incentive Dynamics Engine (IDE) rollout
- Tokenomics overhaul: 50% reduction in circulating supply planned
- Mobile Edge Computing focus for low-latency AI/ML

**Sources:**
- [io.net Official](https://io.net/)
- [io.net Developer Docs](https://developers.io.net/docs/io-worker)
- [io.net $20M Revenue](https://io.net/blog/io-net-20m-in-annualized-on-chain-revenue)

---

### 4. Aethir

**Website:** https://aethir.com/

**Overview:**
Aethir is an enterprise-focused decentralized GPU cloud primarily targeting data centers and professional operators with high-end GPUs. Best suited for operators with 8+ GPU nodes.

**Supported Hardware:**
- Enterprise: H100, H200, A100, GB200, B200, B300
- Consumer: "High-end consumer chips" supported but enterprise GPUs preferred
- **Apple Silicon: NOT SUPPORTED** (NVIDIA focus)

**Minimum Requirements:**
- Compatible GPU setup (H100 preferred for profitability)
- Stable internet connection
- Proper cooling infrastructure
- KYC verification required
- Staking commitment required
- Enterprise SLA adherence
- Application approval required

**Earnings Potential:**
- H100s: $1.45-$3.50/hour
- 8-GPU node: $25,000-$40,000/month
- 95%+ GPU utilization rates reported
- Rewards based on GPU quality, usage, and uptime
- ATH token rewards

**Setup Complexity:** High
- Organization info submission
- Hardware inventory documentation
- KYC verification
- Optional operations demo
- Staking activation required
- Enterprise SLA compliance

**Payout Details:**
- Method: ATH tokens (cryptocurrency)
- Rewards based on GPU utilization and uptime
- Professional-grade revenue model

**Credit Offset:** Yes - ATH tokens can be used within the Aethir ecosystem

**Network Size:** 440,000+ GPU containers, 200+ locations, 94 countries

**2026 Roadmap:**
- Q1 2026: Significant Cloud Host onboarding expansion
- Q2 2026: SCR onboarding for institutional AI clients
- H2 2026: ATH SCR-driven expansion, multi-sector AI enterprise adoption
- Goal: More than double compute network by Q1 2026

**Sources:**
- [Aethir Cloud Host Guide](https://ecosystem.aethir.com/blog-posts/step-by-step-guide-onboarding-as-an-aethir-cloud-host)
- [Aethir Revenue Maximization](https://ecosystem.aethir.com/blog-posts/how-to-maximize-revenue-as-an-aethir-gpu-cloud-host)
- [Aethir 2025 Wrap-Up](https://ecosystem.aethir.com/blog-posts/aethirs-2025-wrap-up-decentralized-gpu-cloud-milestones)

---

### 5. Akash Network

**Website:** https://akash.network/

**Overview:**
Akash is a decentralized cloud computing marketplace using Kubernetes. It's general-purpose (CPUs, GPUs, storage) but primarily focused on enterprise-grade GPUs for AI training workloads.

**Supported Hardware:**
- Enterprise: H200, H100, A100 (primary focus)
- Consumer GPUs: "Could become more viable" as market shifts to inference
- **Apple Silicon: NOT SUPPORTED** (Kubernetes/NVIDIA ecosystem)

**Current GPU Pricing (as provider income):**
- H200 SXM5: $1.95-$3.35/hour
- H100 SXM5: $1.18-$2.53/hour
- A100 SXM4: $0.75-$0.80/hour
- Daily average fee per GPU: ~$20 (January 2025 data)

**Earnings Potential:**
- A100 utilization: 91%+ (supply shortage)
- 428% year-over-year growth in usage
- 80%+ utilization heading into 2026
- AKT token rewards

**Setup Complexity:** High
- Kubernetes infrastructure required
- Provider setup on Akash network
- Datacenter-grade requirements for best opportunities
- Staking may be required

**Payout Details:**
- Method: AKT tokens (cryptocurrency)
- On-chain settlement

**Credit Offset:** Yes - AKT tokens can be used to deploy workloads on Akash

**Network Size:** Growing, with critical A100 supply shortage

**2026 Roadmap:**
- Starcluster initiative: Up to $75M raise for 7,200 GB200 GPUs
- Enterprise-grade datacenter "Nodekeepers"
- Hardware expected online late 2025 into early 2026

**Sources:**
- [Akash GPU Pricing](https://akash.network/pricing/gpus/)
- [Akash GPU Provider Incentives Discussion](https://github.com/orgs/akash-network/discussions/448)
- [State of Akash Q3 2025 - Messari](https://messari.io/report/state-of-akash-q3-2025)

---

### 6. GamerHash

**Website:** https://gamerhash.com/

**Overview:**
GamerHash allows gamers to earn passive income by sharing GPU power for cryptocurrency mining and AI workloads. Over 800,000 users participate.

**Supported Hardware:**
- NVIDIA GPUs: Wide support
- AMD GPUs: Supported
- **Apple Silicon: NOT SUPPORTED** (Windows focus)

**Minimum Requirements:**
- Gaming PC with dedicated GPU
- Windows operating system
- Sufficient GPU power for mining/AI tasks

**Earnings Potential:**
- RTX 4090: ~$0.32/hour at 100% load, up to $150-300/month
- RTX 4070 Ti (~70% of 4090 performance): ~$0.22/hour
- Earnings scale with GPU AI performance
- GamerHash AI workloads can increase earnings

**Setup Complexity:** Low
- Download desktop application
- One-click operation
- Automatic selection of most profitable cryptocurrency/workload
- No technical skills required

**Payout Details:**
- Method: GUSD (stablecoin), exchangeable for USD
- Various cryptocurrency options
- Monthly payments

**Credit Offset:** No - earnings are cryptocurrency-based

**Network Size:** 800,000+ users

**Sources:**
- [GamerHash Official](https://gamerhash.com/en)
- [GamerHash Earnings FAQ](https://gamerhash.com/en/faq/gamerhash-ai-en/how-much-can-i-expect-to-earn)
- [GamerHash Hardware Requirements](https://gamerhash.com/en/faq/mining-en/desktop-application/hardware-required)

---

### 7. Hyperlink

**Website:** https://www.hyperlink.org/

**Overview:**
Hyperlink allows you to earn money by renting out CPU and GPU power for AI, Blockchain, and Cloud tasks. Founded in 2020.

**Supported Hardware:**
- CPUs: Supported
- GPUs: Supported
- Platforms: Mac, Windows, Linux
- **Apple Silicon: SUPPORTED** (Mac support confirmed)
- Future plans: Smartphones, tablets, gaming consoles

**Earnings Potential (Claimed):**
- Desktop: $4,300-$34,700/year
- Laptop: $4,300-$13,000/year
- Servers: Higher earnings
- Earnings depend on core count and processing power

**Setup Complexity:** Low
- Desktop app installation
- Background operation
- Can use computer normally while earning

**Payout Details:**
- Methods: PayPal, ACH (US), Bank Wire (international)
- Processing time: 3-5 business days

**Credit Offset:** No

**Caution:** Some users report difficulty getting support responses and payment issues. Due diligence recommended.

**Sources:**
- [Hyperlink Official](https://www.hyperlink.org/)
- [Hyperlink Review](https://gigsdoneright.com/sell-computing-power/)

---

### 8. Q Blocks

**Website:** https://www.qblocks.cloud/

**Overview:**
Q Blocks is a decentralized GPU marketplace for machine learning engineers and data scientists. Hosts earn when their machines are rented.

**Supported Hardware:**
- GPU-powered machines (NVIDIA primary)
- Minimum requirements must be met
- **Apple Silicon: Likely NOT SUPPORTED** (ML/CUDA focus)

**Earnings Potential:**
- Hourly earnings based on machine specs
- Dedicated resources earn more
- Contact support@qblocks.cloud for earning estimates

**Setup Complexity:** Medium
- Software analyzes machine capabilities
- Provides earning estimates
- Containerized instances for security

**Payout Details:**
- Contact platform for specifics

**Credit Offset:** Unknown

**Sources:**
- [Q Blocks Host](https://www.qblocks.cloud/host/)
- [Q Blocks FAQ](https://www.qblocks.cloud/faq)
- [Q Blocks Requirements](https://www.qblocks.cloud/host/requirements)

---

### 9. Fluence

**Website:** https://fluence.network/

**Overview:**
Fluence is a DePIN cloudless computing platform focused on enterprise-grade compute. Primarily works with Tier 3-4 data centers rather than individual consumers.

**Supported Hardware:**
- Enterprise data center GPUs
- Tier 3-4 datacenter infrastructure required
- **Apple Silicon: NOT SUPPORTED** (enterprise DC focus)

**Requirements:**
- Tier IV data centers: 99.995% uptime
- Verified performance through on-chain attestations
- Real-time telemetry monitoring

**Earnings Potential:**
- Platform generates millions in annualized revenue to providers
- GPU rates: $0.32-$2.24/hour
- Verified datacenter rates: $0.94-$2.04/hour
- 25+ million FLT staked on network

**Setup Complexity:** High (data center operators only)

**Payout Details:**
- Method: FLT tokens (cryptocurrency)
- Revenue share model

**Credit Offset:** Yes - FLT tokens can be used within ecosystem

**Sources:**
- [Fluence Decentralized Cloud Guide](https://www.fluence.network/blog/decentralized-cloud-computing-guide/)
- [Fluence GPU Launch](https://www.fluence.network/blog/fluence-launches-global-and-affordable-gpu-compute-for-ai/)
- [Fluence Vision 2026](https://www.fluence.network/blog/fluence-vision-2026/)

---

### 10. RunPod (Partner/Secure Cloud Program)

**Website:** https://runpod.io/

**Overview:**
RunPod offers a partner program for GPU providers, but with strict datacenter-grade requirements.

**Requirements:**
- NVIDIA GPUs: Ampere generation or newer
- ECC main system memory required
- Separate boot and working storage arrays
- Shared storage cluster (RunPod provides licensing)
- PCIe 4.0 x16
- Large NVMe per-GPU
- High-bandwidth networking: 200 Gbps in partner setups
- **Apple Silicon: NOT SUPPORTED** (NVIDIA datacenter focus)

**Setup Complexity:** Very High (enterprise datacenter level)

**Revenue Model:**
- Hourly GPU usage fees: $0.40-$4.18/hour
- Platform serves 500,000+ developers
- $120M+ ARR (2026)

**Sources:**
- [RunPod Partner Requirements](https://docs.runpod.io/hosting/partner-requirements)
- [RunPod 2026 Statistics](https://fueler.io/blog/runpod-usage-revenue-valuation-growth-statistics)

---

## Apple Silicon Specific Options

### Exo (Self-Hosting Only)

**Website:** https://github.com/exo-explore/exo

**Overview:**
Exo is an open-source tool for running distributed AI inference across consumer devices, including Apple Silicon. It does NOT provide earnings but enables cost savings by self-hosting.

**Supported Hardware:**
- Apple Silicon (M1/M2/M3/M4 series)
- iPhones, iPads, Android devices
- NVIDIA GPUs
- Raspberry Pi
- Any device with compute capability

**Key Features:**
- Peer-to-peer architecture (no master-worker)
- Built on Apple's MLX framework
- Supports models up to 671B parameters across local network
- RDMA over Thunderbolt 5: 80Gb/s, 5-9 microsecond latency
- 38,000+ GitHub stars

**2025-2026 Developments:**
- Exo 1.0 released in cooperation with Apple
- macOS 26.2 integration
- MLX Distributed Framework support
- Trillion-parameter model demonstrations on Thunderbolt 5 Mac clusters

**Earnings Potential:** None directly - cost savings only
**Credit Offset:** N/A - self-hosted solution

**Supported Models:**
- LLaMA (MLX and tinygrad)
- Mistral
- LlaVA
- Qwen
- Deepseek

**Sources:**
- [Exo GitHub](https://github.com/exo-explore/exo)
- [Exo 1.0 and Thunderbolt 5](https://www.geeky-gadgets.com/thunderbolt-5-rdma-macs/)
- [Apple $730K Infrastructure Discount](https://medium.com/@castrojulio/apple-just-gave-everyone-a-730-000-discount-on-ai-infrastructure-and-almost-nobody-noticed-b0ad9c84a566)

### dnet (Distributed Network for Apple Silicon)

**Website:** https://github.com/firstbatchxyz/dnet

**Overview:**
dnet is a distributed LLM inference solution specifically for Apple Silicon clusters. Runs LLMs across Apple Silicon devices with automatic device profiling.

**Features:**
- Modular execution strategies
- Automatic device profiling
- Drop-in OpenAI API compatibility
- Runs models exceeding total cluster memory via compute/I/O overlap
- Built on MLX, inspired by Exo

**Earnings Potential:** None - self-hosting tool only

**Sources:**
- [dnet GitHub](https://github.com/firstbatchxyz/dnet)

---

## Comparison by Hardware Type

### For Apple Silicon (Mac Studio, MacBook Pro M-series)

| Option | Earnings | Notes |
|--------|----------|-------|
| Hyperlink | Claimed $4,300-34,700/yr | Mac support confirmed, but reliability concerns |
| Exo | None (cost savings) | Best for self-hosting inference |
| dnet | None (cost savings) | Apple Silicon cluster tool |

**Reality Check:** No major GPU rental marketplace currently supports Apple Silicon for earning. The MLX/Metal ecosystem is not compatible with CUDA-based platforms that dominate the market.

### For Consumer NVIDIA GPUs (RTX 4090, RTX 3090)

| Platform | Est. Monthly | Difficulty | Payout |
|----------|--------------|------------|--------|
| Vast.ai | $500-1,500 | Medium | PayPal/Wise |
| Salad | $30-180 | Low | PayPal/Gift Cards |
| io.net | Varies | Medium | IO Token |
| GamerHash | $150-300 | Low | GUSD |
| Q Blocks | Varies | Medium | TBD |

### For Professional GPUs (A6000, RTX 6000 Ada)

| Platform | Notes |
|----------|-------|
| Vast.ai | Supported, higher rates |
| RunPod Partner | Strict DC requirements |
| Aethir | Enterprise focus, high returns |

### For Enterprise GPUs (A100, H100, H200)

| Platform | Est. Monthly | Notes |
|----------|--------------|-------|
| Aethir | $25,000-40,000/8-GPU | KYC required, enterprise SLAs |
| Akash | ~$600/GPU | AKT token rewards |
| io.net | Varies | IO token rewards |
| Fluence | Revenue share | DC operators only |

---

## Platforms with Credit Offset (Earn to Use)

These platforms allow you to earn credits/tokens that can directly offset inference or compute costs:

1. **io.net** - IO tokens earned from providing compute can pay for compute usage
2. **Aethir** - ATH tokens usable within ecosystem
3. **Akash** - AKT tokens can deploy workloads
4. **Fluence** - FLT tokens for ecosystem usage

**Note:** Traditional platforms like Vast.ai and Salad pay in cash/gift cards, not platform credits.

---

## Key Takeaways

1. **Apple Silicon Gap:** No major earning platform supports Apple Silicon. Exo/dnet provide self-hosting cost savings only.

2. **Consumer GPU Sweet Spot:** RTX 4090/3090 owners have multiple options:
   - Salad for simplicity ($30-180/mo)
   - Vast.ai for higher returns ($500-1,500/mo at good utilization)
   - io.net for crypto-native earnings

3. **Enterprise Opportunity:** Data center operators with H100s can earn $25,000-40,000/month per 8-GPU node on Aethir.

4. **DePIN Platforms** (io.net, Akash, Aethir, Fluence) offer credit offset capabilities but require crypto/token management.

5. **Setup Complexity Spectrum:**
   - Low: Salad, GamerHash (Windows app)
   - Medium: Vast.ai (Ubuntu), io.net (Docker)
   - High: Aethir (KYC/staking), Akash (Kubernetes), RunPod (DC-grade)

---

## Research Date
February 1, 2026

## Disclaimer
Earnings estimates are based on platform claims and user reports. Actual earnings depend on hardware specifications, market demand, utilization rates, electricity costs, and platform conditions. Cryptocurrency-based earnings are subject to market volatility.
