# Mac Studio Unified Memory Multi-Model Inference

How to leverage Apple Silicon's unified memory architecture to run multiple large language models simultaneously.

## The Unified Memory Advantage

Unlike traditional GPUs with separate VRAM, Apple Silicon uses **unified memory** shared between CPU and GPU:
- **Mac Studio with M1/M2 Ultra**: Up to 128GB or 192GB of unified memory
- **No VRAM limitations**: Models can use as much memory as available
- **Multiple models simultaneously**: Can load many models at once
- **High-quality models**: Enough memory for fp16 models that wouldn't fit on most GPUs

## Key Capabilities

### What 128GB Enables

**Multiple Large Models Concurrently**:
- 2-3× 70B models at q8_0 (~40GB each)
- 4-6× 30B models at fp16 (~60GB total)
- 10+× 7B models at fp16 (~20GB each)
- Mix of different sizes strategically

**Higher Quality Models**:
- Run fp16 instead of quantized versions
- Better quality outputs
- Larger context windows without memory pressure

**Backup/Overflow Capacity**:
- When GPU cluster is saturated, Mac handles overflow
- Complementary to GPU-based inference
- Higher latency but available capacity

## Multi-Instance Ollama Setup

### Why Multiple Ollama Instances

Similar to multi-GPU setups, one Ollama instance = one model at a time:
- **Single instance**: Can only process one model's requests at a time
- **Multiple instances**: Each can handle different models concurrently

### Basic 4-Instance Setup

```bash
#!/bin/bash
# mac_ollama_multi_start.sh

set -e

NUM_INSTANCES=4
BASE_PORT=11434
OLOL_BASE_PORT=50051

echo "Starting $NUM_INSTANCES Ollama instances on Mac Studio..."

# Kill existing instances
pkill -f "ollama serve" || true
pkill -f "olol server" || true
sleep 2

# Start Ollama instances
for ((i=0; i<NUM_INSTANCES; i++)); do
    PORT=$((BASE_PORT + i))

    echo "Starting Ollama instance $i on port $PORT..."
    nohup ollama serve --host 0.0.0.0 --port $PORT \
        > "ollama_mac${i}.log" 2>&1 &

    echo "  Instance $i PID: $!"
done

sleep 10

# Start OLOL servers
for ((i=0; i<NUM_INSTANCES; i++)); do
    OLLAMA_PORT=$((BASE_PORT + i))
    OLOL_PORT=$((OLOL_BASE_PORT + i))

    echo "Starting OLOL server for instance $i..."
    nohup olol server \
        --host 0.0.0.0 \
        --port $OLOL_PORT \
        --ollama-host "http://localhost:$OLLAMA_PORT" \
        > "olol_mac${i}.log" 2>&1 &
done

echo ""
echo "Mac Studio setup complete!"
echo "Instances: 4"
echo "Ollama ports: 11434-11437"
echo "OLOL ports: 50051-50054"
```

### Memory Allocation Strategy

With 128GB total memory:

#### Conservative Allocation (~25GB per instance)
```bash
# 4 instances × 25GB = 100GB, leaving 28GB for system
Instance 0: ~25GB - Large quality model (llama3:70b-q8_0)
Instance 1: ~25GB - Specialized model (codellama:34b-fp16)
Instance 2: ~25GB - Fast medium model (qwen2.5:32b-q8_0)
Instance 3: ~25GB - General purpose (mistral:22b-fp16)
```

#### Aggressive Allocation (~30GB per instance)
```bash
# 4 instances × 30GB = 120GB, leaving 8GB for system
Instance 0: ~30GB - Flagship (llama3:70b-fp16)
Instance 1: ~30GB - Coding specialist (codellama:70b-q4_0)
Instance 2: ~30GB - Reasoning (qwen2.5:72b-q4_0)
Instance 3: ~30GB - General (mixtral:8x22b-q4_0)
```

## Strategic Model Loading

### Load Different Model Types

```bash
#!/bin/bash
# load_mac_models.sh

# Instance 0 - Large flagship model
curl http://localhost:11434/api/pull -d '{"name":"llama3:70b-instruct-q8_0"}' &

# Instance 1 - Coding specialist
curl http://localhost:11435/api/pull -d '{"name":"codellama:34b-instruct-fp16"}' &

# Instance 2 - Reasoning/analysis
curl http://localhost:11436/api/pull -d '{"name":"qwen2.5:32b-instruct-fp16"}' &

# Instance 3 - Fast general purpose
curl http://localhost:11437/api/pull -d '{"name":"mistral:22b-instruct-fp16"}' &

wait
echo "All models loaded!"
```

### Dynamic Model Management

```bash
# Check what's loaded
for port in 11434 11435 11436 11437; do
    echo "Port $port:"
    curl -s "http://localhost:$port/api/tags" | jq '.models[].name'
done

# Unload a model to free memory
curl http://localhost:11434/api/delete -d '{"name":"old-model:tag"}'

# Load new model
curl http://localhost:11434/api/pull -d '{"name":"new-model:tag"}'
```

## Integration with GPU Cluster

### Mac as Coordinator + Worker

The Mac Studio serves dual role:
1. **Coordinator**: Runs OLOL proxy to manage entire cluster
2. **Worker**: Participates in inference with its own instances

```bash
#!/bin/bash
# mac_proxy_and_worker.sh

set -e

VM1_IP="192.168.1.10"
VM2_IP="192.168.1.11"
NUM_MAC_INSTANCES=4

# Start local Ollama instances (as shown above)
# ...

# Build server list including Mac instances
SERVERS=""

# Add VM1 servers (5× M4000)
for i in {0..4}; do
    SERVERS="${SERVERS}${VM1_IP}:$((50051 + i)),"
done

# Add VM2 servers (3× mixed GPUs)
for i in {0..2}; do
    SERVERS="${SERVERS}${VM2_IP}:$((50051 + i)),"
done

# Add local Mac instances
for i in {0..3}; do
    SERVERS="${SERVERS}localhost:$((50051 + i)),"
done

# Remove trailing comma
SERVERS=${SERVERS%,}

# Start proxy
nohup olol proxy \
    --host 0.0.0.0 \
    --port 8000 \
    --servers "$SERVERS" \
    --distributed \
    --discovery \
    > olol_proxy.log 2>&1 &

echo "OLOL cluster with Mac Studio active!"
echo "Total compute instances: 12"
echo "  - 5× M4000 GPUs"
echo "  - 3× Mixed GPUs"
echo "  - 4× Mac Studio instances"
```

## Monitoring Mac Performance

### Memory Usage

```bash
# Check overall memory
vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//' | awk '{print $1 * 4096 / 1024 / 1024 / 1024 " GB"}'

# Monitor continuously
while true; do
    clear
    echo "=== Mac Memory Usage ==="
    vm_stat | grep -E "Pages (free|active|inactive|wired)"
    sleep 5
done
```

### Process-Level Monitoring

```bash
# Check Ollama memory usage
ps aux | grep "ollama serve" | awk '{print $2}' | xargs -I {} sh -c 'echo "PID: {}"; ps -p {} -o rss='

# More detailed view
top -pid $(pgrep ollama | head -1)
```

### Activity Monitor

Use macOS Activity Monitor:
- Filter for "ollama"
- Sort by Memory
- Watch Real Memory column

## Workload Distribution Strategies

### Strategy 1: Mac for Large/Quality, GPUs for Speed

```bash
# Mac Studio: Large, high-quality models
Mac instances: llama3:70b-fp16, codellama:34b-fp16

# GPU cluster: Smaller, faster models
GPU instances: mistral:7b-q8_0, llama3:8b-q8_0
```

**When requests come in**:
- Small/fast requests → GPUs
- Large/quality requests → Mac
- OLOL automatically routes based on model

### Strategy 2: Mac for Overflow/Backup

```bash
# Primary: GPU cluster handles most work
# Backup: Mac handles overflow when GPUs saturated
```

**Benefits**:
- GPUs provide low latency
- Mac ensures throughput never drops
- Automatic failover if GPUs busy

### Strategy 3: Specialized Models on Mac

```bash
# Mac: Specialized/rarely-used models
Mac: medical-model:70b, legal-model:34b, code-specific:34b

# GPUs: Common general-purpose models
GPUs: mistral:7b, llama3:8b (highly duplicated)
```

**Benefits**:
- GPUs stay focused on high-volume tasks
- Mac handles niche requests
- Better overall resource allocation

## Performance Characteristics

### Latency Expectations

| Model Size | Mac Studio M2 Ultra | NVIDIA A4000 16GB |
|-----------|---------------------|-------------------|
| 7B fp16   | ~30 tokens/sec      | ~60 tokens/sec    |
| 13B fp16  | ~20 tokens/sec      | ~40 tokens/sec    |
| 30B fp16  | ~10 tokens/sec      | ~20 tokens/sec    |
| 70B q8_0  | ~5 tokens/sec       | Won't fit         |

**Key Insight**: Mac is slower per token, but can run models GPUs can't fit.

### Throughput Optimization

For maximum throughput:
```bash
# More instances (if memory allows)
NUM_INSTANCES=6  # 6× ~20GB models

# Lighter quantization
# q4_0 instead of fp16 means more models loaded
```

## Common Patterns

### Pattern 1: Quality-First

```bash
# All fp16 models for maximum quality
# Accept lower throughput
llama3:70b-fp16, mistral:22b-fp16, qwen2.5:32b-fp16
```

### Pattern 2: Quantity-First

```bash
# More instances with q4_0/q5_0
# Higher throughput, lower quality
6-8 instances with 13B-30B models at q4_0
```

### Pattern 3: Mixed Strategy

```bash
# 1-2 fp16 flagship models for critical requests
# 2-3 q8_0 models for balanced quality/speed
Instance 0: llama3:70b-fp16      # Critical quality
Instance 1: mistral:22b-fp16     # High quality
Instance 2: qwen2.5:32b-q8_0     # Balanced
Instance 3: codellama:34b-q5_0   # More throughput
```

## Troubleshooting

### System Running Slow

**Problem**: macOS feels sluggish

**Solutions**:
1. Reduce number of instances
2. Use lighter quantization
3. Leave more free memory (>16GB)
4. Close other applications

### Out of Memory

**Problem**: "Cannot allocate memory" errors

**Solutions**:
```bash
# Check memory pressure
memory_pressure

# Unload some models
curl http://localhost:11434/api/delete -d '{"name":"large-model"}'

# Restart with fewer instances
```

### Thermal Throttling

**Problem**: Performance degrades over time

**Solutions**:
1. Improve airflow around Mac Studio
2. Reduce concurrent instances during hot periods
3. Use lighter quantization (less compute = less heat)
4. Monitor with: `sudo powermetrics --samplers smc`

## Best Practices

1. **Leave headroom**: Don't use all 128GB, leave 10-20GB free
2. **Monitor memory pressure**: Check `memory_pressure` regularly
3. **Load strategically**: Pre-load frequently used models
4. **Use appropriate quantization**: fp16 only when quality matters
5. **Scale instances based on workload**: More instances ≠ better performance
6. **Consider thermal limits**: Sustained high load may throttle

## Summary

**Mac Studio Strengths**:
- Massive unified memory for large models
- Multiple models simultaneously
- High quality (fp16) models that don't fit on GPUs
- Excellent for specialized/overflow capacity

**Best Used For**:
- Large model inference (70B+)
- High-quality fp16 inference
- Specialized models with lower request volume
- Backup/overflow capacity in mixed cluster

**Not Ideal For**:
- Maximum tokens/second (GPUs faster)
- Ultra-low latency requirements
- Extremely high throughput tasks

When combined with GPU cluster, Mac Studio provides flexible, high-capacity inference capability that complements GPU speed with memory abundance.
