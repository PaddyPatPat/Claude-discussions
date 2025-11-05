# OLOL Distributed Inference Setup

OLOL (Ollama-to-Ollama) is a distributed inference system that enables coordinating multiple Ollama instances across different machines to maximize GPU utilization and throughput.

**Project**: https://github.com/K2/olol

## What OLOL Provides

### Core Capabilities

1. **Distributed Inference**: Automatically splits large models across multiple servers using layer partitioning
2. **Load Balancing**: Distributes requests across available servers to maximize throughput
3. **Model-Aware Routing**: Routes requests to servers that have the requested model loaded
4. **Auto-Discovery**: Automatically detects and uses distributed inference for large models (13B+)
5. **API Compatibility**: Maintains full Ollama API compatibility

### Key Features

- **Layer Partitioning**: Automatically splits model layers across available servers
- **Intelligent Quantization**: Automatically selects the best quantization level based on hardware and model size
- **Concurrent Processing**: Handle multiple inference requests simultaneously across distributed resources
- **Monitoring**: Built-in status endpoints for cluster health monitoring

## Installation Options

```bash
# Install from PyPI
uv pip install olol

# Install with extras
uv pip install "olol[proxy,async]"
```

## Architecture Patterns

### Pattern 1: Single Ollama per Machine (Simplest)

Best for: Initial setup, testing, machines with limited GPUs

```bash
# On each compute node
olol server --host 0.0.0.0 --port 50051 --ollama-host http://localhost:11434

# On coordination machine
olol proxy --host 0.0.0.0 --port 8000 --servers "node1:50051,node2:50051" --distributed --discovery
```

**Pros**:
- Simplest setup
- Let Ollama handle multi-GPU coordination within each machine
- OLOL handles cross-machine distribution

**Cons**:
- May not maximize multi-GPU utilization on a single machine
- One Ollama instance = one model at a time per machine

### Pattern 2: Multiple Ollama per Machine (Maximum Utilization)

Best for: Batch processing, maximizing GPU utilization, concurrent small models

```bash
# On machine with multiple GPUs
CUDA_VISIBLE_DEVICES=0 ollama serve --port 11434 &
CUDA_VISIBLE_DEVICES=1 ollama serve --port 11435 &
CUDA_VISIBLE_DEVICES=2 ollama serve --port 11436 &

olol server --port 50051 --ollama-host http://localhost:11434 &
olol server --port 50052 --ollama-host http://localhost:11435 &
olol server --port 50053 --ollama-host http://localhost:11436 &

# Proxy coordinates all instances
olol proxy --port 8000 --servers "machine:50051,machine:50052,machine:50053"
```

**Pros**:
- Each GPU can run its own model instance
- Maximum hardware utilization for concurrent requests
- Each GPU stays busy with independent work

**Cons**:
- More complex setup
- More services to monitor
- Higher memory overhead from multiple Ollama processes

### Pattern 3: Hybrid Approach (Flexible)

Best for: Mixed workloads (concurrent small models + occasional large models)

```bash
# GPUs 0-1: Individual instances for small models
CUDA_VISIBLE_DEVICES=0 ollama serve --port 11434 &
CUDA_VISIBLE_DEVICES=1 ollama serve --port 11435 &

# GPU 2: Part of distributed inference cluster
olol rpc-server --host 0.0.0.0 --port 50054 --device cuda

# Proxy with both standard and RPC servers
olol proxy --servers "machine:50051,machine:50052" --rpc-servers "machine:50054"
```

**Pros**:
- Flexibility for different workload types
- Can handle both concurrent small models and distributed large models

**Cons**:
- Most complex setup
- Requires understanding of when to use which mode

## Distributed Inference for Large Models

### When to Use Distributed Inference

- Models larger than 13B parameters
- Models that don't fit in a single GPU's VRAM
- When you want to parallelize computation across GPUs for speed

### Setup for Distributed Inference

```bash
# On each compute node with GPUs
olol rpc-server --host 0.0.0.0 --port 50052 --device cuda --flash-attention --context-window 16384

# Proxy coordinates distributed inference
olol proxy --servers "node1:50051,node2:50051" --distributed --rpc-servers "node1:50052,node2:50052"
```

### How It Works

1. **Automatic Detection**: OLOL detects when a model is too large for a single node
2. **Layer Distribution**: Model layers are automatically distributed across available RPC servers
3. **Parallel Execution**: Inference runs across multiple GPUs simultaneously
4. **Result Aggregation**: Final output is assembled and returned to the client

## Monitoring and Management

### Check Cluster Status

```bash
curl http://localhost:8000/api/status | jq
```

### Monitor GPU Utilization

```bash
# Continuous monitoring
watch -n 1 nvidia-smi

# Log to file for analysis
while true; do
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv >> gpu_usage.log
    sleep 5
done
```

### Per-GPU Detailed Monitoring

```bash
nvidia-smi dmon -s pucvmet -d 1
```

## Use Cases

### Batch Inference (High Throughput)
- Many small requests processed concurrently
- Each GPU handles different requests independently
- Maximize GPU utilization across all hardware

### Large Model Inference
- Models too big for single GPU
- Distributed across multiple GPUs
- Trade-off: slightly higher latency for ability to run larger models

### Mixed Workloads
- Combination of small and large models
- Dynamic routing based on model requirements
- Flexible resource allocation

## Performance Optimization

### For Maximum Throughput
- Use Pattern 2 (Multiple Ollama per Machine)
- Load different quantization levels on different GPUs
- Keep queue depth high to ensure GPUs always have work

### For Large Models
- Use distributed inference with RPC servers
- Enable flash attention for faster processing
- Increase context window as needed

### For Reliability
- Use auto-discovery to handle node failures
- Monitor cluster status regularly
- Implement request retry logic in client code

## Common Pitfalls

1. **Single Ollama + Multiple GPUs = Idle GPUs**: One Ollama instance typically only uses one GPU
2. **Not Enough Concurrent Requests**: GPUs idle if queue is too shallow
3. **Mismatched Model Sizes**: Loading models too large for GPU VRAM causes failures
4. **Network Bottlenecks**: Ensure good network connectivity between nodes for distributed inference

## Next Steps

1. Start with Pattern 1 for initial testing
2. Measure baseline performance
3. Scale to Pattern 2 for production batch processing
4. Add distributed inference capability for large models as needed
