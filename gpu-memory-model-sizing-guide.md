# GPU Memory and Model Sizing Guide

A practical guide for determining which LLM models fit on which GPUs based on VRAM capacity and quantization levels.

## Quick Reference Table

| GPU VRAM | Recommended Models (q8_0) | Aggressive (q4_0/q5_0) | Full Precision (fp16) |
|----------|---------------------------|------------------------|----------------------|
| 8GB      | 7B models                 | 13B models             | 3B models            |
| 12GB     | 13B models                | 20B models             | 7B models            |
| 16GB     | 20B models                | 30B models             | 13B models           |
| 24GB     | 30B models                | 70B models (tight)     | 20B models           |
| 48GB+    | 70B models                | 405B (distributed)     | 30B+ models          |

## Understanding Quantization

Quantization reduces model size by representing weights with fewer bits, trading some quality for memory efficiency.

### Common Quantization Levels

- **fp16** (Full Precision): ~2 bytes per parameter, highest quality
- **q8_0**: ~1 byte per parameter, minimal quality loss
- **q5_0/q5_1**: ~0.625 bytes per parameter, good balance
- **q4_0/q4_1**: ~0.5 bytes per parameter, noticeable but acceptable quality loss
- **q3_0**: ~0.375 bytes per parameter, significant quality degradation

### Size Calculation Formula

```
Approximate VRAM needed = (Parameters × Bytes per parameter) + Overhead

Examples:
- 7B model at q8_0: (7B × 1 byte) + ~1.5GB overhead ≈ 8.5GB
- 13B model at q5_0: (13B × 0.625 bytes) + ~1.5GB overhead ≈ 9.6GB
- 30B model at q4_0: (30B × 0.5 bytes) + ~1.5GB overhead ≈ 16.5GB
```

**Note**: Overhead includes context buffer, activations, and KV cache

## Specific GPU Recommendations

### 8GB VRAM (e.g., RTX 3070, M4000)

**Optimal**:
- 7B models at q8_0 (highest quality for this size)
- Examples: `mistral:7b-instruct-q8_0`, `llama3:8b-q8_0`

**Stretch**:
- 13B models at q4_0 (lower quality but larger model)
- 10B models at q5_0 (good balance)

**Avoid**:
- Anything larger than 13B
- fp16 models over 3B parameters

### 12GB VRAM (e.g., RTX 3060, A2000)

**Optimal**:
- 13B models at q8_0
- 7B models at fp16 (if highest quality needed)

**Stretch**:
- 20B models at q5_0
- 27B models at q4_0

**Avoid**:
- 30B+ models (even at q4_0)
- fp16 models over 7B

### 16GB VRAM (e.g., A4000, RTX 4000)

**Optimal**:
- 20B models at q8_0
- 13B models at fp16

**Stretch**:
- 30B models at q4_0/q5_0
- 27B models at q8_0

**Avoid**:
- 70B models (even heavily quantized)
- fp16 models over 13B

### 24GB VRAM (e.g., RTX 3090, A5000)

**Optimal**:
- 30B models at q8_0
- 20B models at fp16

**Stretch**:
- 70B models at q4_0 (very tight)
- 50B models at q5_0

**Good for**:
- Running multiple smaller models
- Two 13B models simultaneously

## Strategic Model Placement Example

### Multi-GPU Setup Configuration

**5× 8GB GPUs (M4000)**:
```bash
# Strategy: Multiple concurrent 7B models for throughput
GPU 0: mistral:7b-instruct-q8_0
GPU 1: llama3:8b-q8_0
GPU 2: qwen2:7b-q8_0
GPU 3: codellama:7b-q8_0
GPU 4: neural-chat:7b-q8_0
```

**1× 12GB GPU (RTX 3060)**:
```bash
# Strategy: Mid-size model at good quality
GPU: mistral:7b-instruct-fp16  # or llama3:13b-q8_0
```

**1× 12GB GPU (A2000)**:
```bash
# Strategy: Specialized model
GPU: codellama:13b-q8_0  # or qwen2:14b-q5_0
```

**1× 16GB GPU (A4000)**:
```bash
# Strategy: Larger model or highest quality
GPU: qwen2.5:14b-q8_0  # or mistral:7b-fp16 for highest quality
```

**Mac Studio (128GB Unified Memory)**:
```bash
# Strategy: Large models at high quality
Instance 0: llama3:70b-q8_0    (~40GB)
Instance 1: codellama:34b-fp16 (~68GB)
Instance 2: mistral:22b-fp16   (~44GB)
Instance 3: qwen2.5:32b-q8_0   (~35GB)
# Total: ~187GB if all loaded, but can strategically load/unload
```

## Context Window Considerations

Larger context windows require more VRAM. The overhead grows with:
- Context length (2k, 4k, 8k, 16k, 32k tokens)
- Model size (larger models have larger hidden states)

### Context Window Memory Impact

| Model Size | 2k context | 8k context | 32k context |
|------------|------------|------------|-------------|
| 7B         | +0.5GB     | +1.5GB     | +4GB        |
| 13B        | +1GB       | +2.5GB     | +8GB        |
| 30B        | +2GB       | +5GB       | +16GB       |

**Practical Implication**: If you need large context windows, reduce model size or quantization level.

## Batch Size Considerations

For batch inference, each additional sequence in the batch requires memory:

```
Memory per batch item ≈ (context_length × hidden_size × 2 bytes)

Example for 7B model with 4k context:
- Single sequence: ~1.5GB overhead
- Batch size 4: ~6GB overhead
- Batch size 8: ~12GB overhead
```

**For batch processing**: Keep batch size low (1-2) on smaller GPUs, higher on larger GPUs.

## Real-World Examples

### Example 1: Maximizing Throughput (Many Small Requests)

**Goal**: Process thousands of short inference requests quickly

**Configuration**:
- Load 7B models at q8_0 on all 8GB GPUs
- Each GPU handles independent requests
- High concurrency, low latency

### Example 2: Quality-Focused (Few High-Quality Responses)

**Goal**: Best possible quality for smaller number of requests

**Configuration**:
- Load 7B models at fp16 on 12GB GPUs
- Or load 13B models at fp16 on Mac Studio
- Lower concurrency, higher quality

### Example 3: Large Model Capability

**Goal**: Run 70B+ models that don't fit on single GPU

**Configuration**:
- Use distributed inference across multiple GPUs
- Or run on Mac Studio with 128GB unified memory
- Quantize to q4_0 or q5_0 to fit

## Model Loading Strategies

### Pre-Loading (Recommended for Batch Processing)

```bash
# Load models before starting inference
for model in mistral:7b llama3:8b qwen2:7b; do
    curl http://localhost:11434/api/pull -d "{\"name\":\"$model\"}"
done
```

**Pros**: Fast inference startup, predictable performance
**Cons**: Uses VRAM even when idle

### On-Demand Loading

```bash
# Models load automatically when first requested
# No pre-loading needed
```

**Pros**: Efficient VRAM usage
**Cons**: First request has long latency, unpredictable load times

### Mixed Strategy

```bash
# Pre-load frequently used models
# Let rarely-used models load on-demand
```

## Memory Monitoring

### Check Current Usage

```bash
# See what's loaded
nvidia-smi

# Per-GPU breakdown
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
```

### Continuous Monitoring

```bash
# Real-time updates
watch -n 1 nvidia-smi

# Log to file
nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv -l 5 >> vram_usage.log
```

## Troubleshooting

### Out of Memory Errors

**Solutions**:
1. Use smaller model (30B → 13B → 7B)
2. Increase quantization (q8_0 → q5_0 → q4_0)
3. Reduce context window
4. Lower batch size
5. Use distributed inference across multiple GPUs

### Model Too Slow

**Solutions**:
1. Decrease quantization (q4_0 → q5_0 → q8_0) - uses more VRAM but faster
2. Use flash attention if supported
3. Reduce context window
4. Use smaller model
5. Distribute across more GPUs

### Uneven GPU Utilization

**Solutions**:
1. Use multiple Ollama instances per machine
2. Balance model sizes across GPUs
3. Ensure enough concurrent requests
4. Check network bottlenecks for distributed setups

## Recommendations Summary

**For Learning/Experimentation**:
- Start with 7B models at q8_0
- Good quality, fits on most GPUs
- Fast inference

**For Production Batch Processing**:
- 7B-13B models at q8_0 for balance
- Maximize concurrent instances
- Pre-load models

**For Quality-Critical Applications**:
- Use fp16 when possible
- Accept lower throughput
- Consider Mac Studio for large fp16 models

**For Cost Optimization**:
- Aggressive quantization (q4_0)
- Maximize GPU utilization
- Use distributed inference for large models
