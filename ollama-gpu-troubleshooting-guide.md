# Ollama GPU Troubleshooting Guide

A comprehensive guide for diagnosing and resolving GPU-related issues with Ollama, focusing on why models might use CPU instead of GPU and how to fix it.

## Symptom: Ollama Using CPU Instead of GPU

When you load an LLM in Ollama and run inference, but `nvidia-smi` shows no GPU utilization, the model is running on CPU. This guide helps diagnose and resolve the issue.

## Prerequisites Verification

### 1. Verify GPUs Are Working

```bash
# Check GPU visibility
nvidia-smi

# Should show all your GPUs with their status
# Look for "No running processes found" initially
```

**Expected output**: List of GPUs with temperature, memory usage, and utilization stats.

### 2. Check CUDA Installation

```bash
# Check CUDA compiler version
nvcc --version

# If command not found, CUDA toolkit is not installed
```

**If CUDA toolkit is missing**, install it:

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nvidia-cuda-toolkit

# Verify installation
nvcc --version
```

**Important**: Having `nvidia-smi` does **not** mean CUDA toolkit is installed. `nvidia-smi` comes with the driver, but CUDA toolkit is separate and required for GPU compute.

### 3. Verify Ollama Installation

```bash
# Check Ollama version
ollama --version

# Ensure it's a recent version with GPU support
```

**If Ollama is outdated or installed via snap**, reinstall with official installer:

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

## Diagnostic Steps

### 1. Enable Debug Mode

Debug mode shows detailed information about GPU detection and initialization:

```bash
# Stop any running Ollama service first
sudo systemctl stop ollama

# Start in debug mode
OLLAMA_DEBUG=1 ollama serve
```

**Look for in the output**:
- GPU detection messages
- CUDA initialization
- Memory allocation
- Model layer distribution

### 2. Common Debug Output Issues

**"listen tcp 127.0.0.1:11434: bind: address already in use"**

This means Ollama server is already running.

**Solutions**:

```bash
# Option 1: Stop the service
sudo systemctl stop ollama

# Option 2: Kill the process
pkill ollama

# Option 3: Kill process using the port
sudo lsof -ti:11434 | xargs sudo kill -9

# Option 4: Use a different port
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_DEBUG=1 ollama serve
```

**Verify port is free**:
```bash
sudo netstat -tlnp | grep 11434
# or
sudo ss -tlnp | grep 11434
```

### 3. Understanding Ollama's Architecture

**Key Components**:
- `ollama serve` - Starts the server daemon (persistent)
- `ollama run model` - Client that connects to server
- `/bye` - Ends the chat session only, **not the server**

**Important**: The server typically runs as a background systemd service and persists between terminal sessions. Closing your terminal or tmux session doesn't stop the server.

**Check if Ollama is running as a service**:
```bash
sudo systemctl status ollama

# View service logs
journalctl -u ollama -f
```

## Configuration for GPU Usage

### Environment Variables

Set these before starting Ollama server:

```bash
# Make all GPUs visible to Ollama
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

# Specify number of GPUs to use
export OLLAMA_NUM_GPU=5

# Reduce GPU memory overhead (optional)
export OLLAMA_GPU_OVERHEAD=0

# Specify number of layers on GPU (-1 = auto-detect)
export OLLAMA_GPU_LAYERS=-1

# Start server with these settings
OLLAMA_DEBUG=1 ollama serve
```

### Per-Session Configuration

```bash
# Start Ollama with specific GPU configuration
OLLAMA_NUM_GPU=5 CUDA_VISIBLE_DEVICES=0,1,2,3,4 OLLAMA_DEBUG=1 ollama serve
```

## Monitoring GPU Usage

### Real-Time Monitoring

```bash
# Update every 0.5 seconds
watch -n 0.5 nvidia-smi

# Or continuously in a loop
while true; do clear; nvidia-smi; sleep 0.5; done
```

**What to look for when model is running**:
- GPU memory usage increases
- GPU utilization percentage > 0%
- Power draw increases
- Temperature may rise

### Detailed GPU Monitoring

```bash
# Show specific metrics
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv -l 1

# Monitor specific GPU
nvidia-smi -i 0 -l 1
```

## Multi-GPU Configuration

### When Models Fit on One GPU

By default, if a model fits on a single GPU, Ollama will use one GPU. This is **optimal** for performance - multi-GPU usage for small models typically **reduces** performance due to:
- Inter-GPU communication overhead
- Memory transfer latency
- Synchronization costs

### Force Multi-GPU Usage

**Using environment variables**:

```bash
# Force Ollama to use 5 GPUs
export OLLAMA_NUM_GPU=5
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

ollama serve
```

**Using Modelfile** (custom configuration):

```bash
# Create a Modelfile
cat > Modelfile << EOF
FROM llama2
PARAMETER num_gpu 5
EOF

# Create model from Modelfile
ollama create my-multi-gpu-model -f Modelfile

# Run the model
ollama run my-multi-gpu-model
```

### When Multi-GPU Helps

Multi-GPU usage is beneficial for:
- **Large models** that don't fit on single GPU
- **Very large context windows** (long conversations)
- **Batch processing** multiple requests simultaneously
- **Models barely fitting** on one GPU (leaves room for context)

### When Single GPU is Better

For models that fit comfortably on one GPU:
- **Faster inference** (no communication overhead)
- **Lower latency** (no inter-GPU synchronization)
- **Simpler setup** (fewer moving parts)
- **Other GPUs available** for other models/workloads

## Common Issues and Solutions

### Issue 1: CUDA Not Found

**Symptoms**:
- Ollama uses CPU
- Debug output shows "CUDA not available"

**Solution**:
```bash
# Install CUDA toolkit
sudo apt update
sudo apt install nvidia-cuda-toolkit

# Verify
nvcc --version
```

### Issue 2: Driver/CUDA Version Mismatch

**Symptoms**:
- GPUs visible with `nvidia-smi`
- CUDA installed
- Still using CPU

**Check compatibility**:
```bash
# Check driver version
nvidia-smi | grep "Driver Version"

# Check CUDA version
nvcc --version
```

**Solution**: Ensure driver version supports your CUDA version. See [NVIDIA CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/) table.

### Issue 3: Insufficient GPU Memory

**Symptoms**:
- Small models use GPU
- Large models fall back to CPU
- "Out of memory" errors in logs

**Solutions**:

```bash
# Check available memory
nvidia-smi

# Try smaller model or quantization
ollama run llama2:7b-q4_0  # Instead of llama2:70b
```

See [gpu-memory-model-sizing-guide.md](gpu-memory-model-sizing-guide.md) for model sizing details.

### Issue 4: Model Format Incompatibility

**Symptoms**:
- Some models use GPU
- Others don't despite fitting in memory

**Cause**: Some older model formats may not support GPU acceleration.

**Solution**: Use models from Ollama's official registry or recent GGUF format.

### Issue 5: Docker GPU Access

If running Ollama in Docker without GPU access:

```bash
# Check current container
docker ps

# Stop container
docker stop ollama

# Start with GPU access
docker run --gpus all -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

## Verification Checklist

After configuration changes, verify GPU usage:

- [ ] CUDA toolkit installed: `nvcc --version`
- [ ] Ollama version recent: `ollama --version`
- [ ] Service stopped: `sudo systemctl stop ollama`
- [ ] Environment variables set: `echo $OLLAMA_NUM_GPU`
- [ ] Debug mode started: `OLLAMA_DEBUG=1 ollama serve`
- [ ] GPU detection in logs: Check for CUDA initialization
- [ ] Model loaded: `ollama run model-name`
- [ ] GPU monitoring active: `watch -n 0.5 nvidia-smi`
- [ ] GPU utilization > 0%: Check nvidia-smi output
- [ ] GPU memory allocated: Check memory usage in nvidia-smi

## Command-Line Options

### Available Ollama Commands

```bash
# Server commands
ollama serve              # Start server
ollama serve --help       # Show server options

# Client commands
ollama list               # List downloaded models
ollama pull model-name    # Download model
ollama run model-name     # Run model (connects to server)
ollama rm model-name      # Remove model
```

**Note**: `ollama run` does **not** accept GPU-specific flags like `--num-gpu` or `--num-gpu-layers`. GPU configuration is done through:
1. Environment variables
2. Modelfile parameters
3. Server configuration

### Testing GPU Configuration

```bash
# Start server with specific GPU config
OLLAMA_NUM_GPU=1 CUDA_VISIBLE_DEVICES=0 ollama serve

# In another terminal, run a model
ollama run llama2

# Monitor GPU 0 specifically
nvidia-smi -i 0 -l 1
```

## Best Practices

### For Development/Testing

```bash
# Use debug mode
OLLAMA_DEBUG=1 ollama serve

# Monitor in separate terminal
watch -n 0.5 nvidia-smi
```

### For Production

```bash
# Run as systemd service
sudo systemctl enable ollama
sudo systemctl start ollama

# Configure via environment in service file
sudo systemctl edit ollama

# Add:
[Service]
Environment="OLLAMA_NUM_GPU=5"
Environment="CUDA_VISIBLE_DEVICES=0,1,2,3,4"

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

### For Multiple Models

Instead of forcing multi-GPU for single model, run different models on different GPUs:

```bash
# See multi-gpu-ollama-setup-guide.md for details

# Terminal 1 - GPU 0
CUDA_VISIBLE_DEVICES=0 ollama serve --host 127.0.0.1:11434

# Terminal 2 - GPU 1
CUDA_VISIBLE_DEVICES=1 ollama serve --host 127.0.0.1:11435

# Terminal 3 - GPUs 2-4
CUDA_VISIBLE_DEVICES=2,3,4 ollama serve --host 127.0.0.1:11436
```

## Quick Diagnostic Script

```bash
#!/bin/bash
# ollama-gpu-check.sh - Quick GPU diagnostic

echo "=== Ollama GPU Diagnostic ==="
echo ""

echo "1. NVIDIA Driver:"
nvidia-smi --version 2>/dev/null || echo "  ❌ nvidia-smi not found"
echo ""

echo "2. CUDA Toolkit:"
nvcc --version 2>/dev/null || echo "  ❌ CUDA toolkit not installed"
echo ""

echo "3. Ollama Version:"
ollama --version 2>/dev/null || echo "  ❌ Ollama not found"
echo ""

echo "4. Ollama Service Status:"
systemctl is-active ollama 2>/dev/null || echo "  Not running as service"
echo ""

echo "5. Port 11434 Status:"
sudo lsof -ti:11434 > /dev/null 2>&1 && echo "  ✓ In use" || echo "  Available"
echo ""

echo "6. GPUs Detected:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "  ❌ No GPUs detected"
echo ""

echo "7. Environment Variables:"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "  OLLAMA_NUM_GPU=${OLLAMA_NUM_GPU:-not set}"
echo "  OLLAMA_GPU_LAYERS=${OLLAMA_GPU_LAYERS:-not set}"
```

Usage:
```bash
chmod +x ollama-gpu-check.sh
./ollama-gpu-check.sh
```

## Related Guides

- [multi-gpu-ollama-setup-guide.md](multi-gpu-ollama-setup-guide.md) - Running multiple Ollama instances per GPU
- [gpu-memory-model-sizing-guide.md](gpu-memory-model-sizing-guide.md) - Matching models to GPU memory
- [olol-distributed-inference-setup.md](olol-distributed-inference-setup.md) - Distributed inference across multiple machines

## Summary

**Most Common Issues**:
1. Missing CUDA toolkit (not just drivers)
2. Ollama service already running (port conflict)
3. Wrong environment variables or not set
4. Model too large for available GPU memory

**Key Diagnostics**:
- `OLLAMA_DEBUG=1 ollama serve` - See what Ollama detects
- `nvidia-smi` while running inference - Verify GPU usage
- `nvcc --version` - Confirm CUDA toolkit installed
- `sudo systemctl status ollama` - Check service status

**GPU Configuration**:
- Use environment variables: `OLLAMA_NUM_GPU`, `CUDA_VISIBLE_DEVICES`
- Configure through Modelfile for custom models
- Single GPU is optimal for models that fit
- Multi-GPU useful for large models or concurrent workloads
