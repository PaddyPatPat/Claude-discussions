# Multi-GPU Ollama Setup Guide

How to run multiple Ollama instances on a single machine to maximize GPU utilization.

## The Problem: Single Ollama = Single GPU

By default, a single Ollama instance uses only one GPU, even if multiple GPUs are available. This means:
- With 5 GPUs and 1 Ollama instance → 4 idle GPUs ❌
- With 5 GPUs and 5 Ollama instances → All GPUs busy ✅

## Solution: Multiple Ollama Instances

Run separate Ollama instances on different ports, each bound to a specific GPU using `CUDA_VISIBLE_DEVICES`.

### Basic Concept

```bash
# GPU 0 - Ollama on port 11434
CUDA_VISIBLE_DEVICES=0 ollama serve --host 0.0.0.0 --port 11434 &

# GPU 1 - Ollama on port 11435
CUDA_VISIBLE_DEVICES=1 ollama serve --host 0.0.0.0 --port 11435 &

# GPU 2 - Ollama on port 11436
CUDA_VISIBLE_DEVICES=2 ollama serve --host 0.0.0.0 --port 11436 &
```

## Startup Scripts for Different Configurations

### Configuration 1: 5× GPUs (e.g., 5× M4000 8GB)

Create `start_ollama_5gpu.sh`:

```bash
#!/bin/bash
# Startup script for 5× GPU Ollama instances

set -e

echo "Starting 5× Ollama instances for 5× GPUs..."

# Configuration
BASE_PORT=11434
OLOL_BASE_PORT=50051

# Kill any existing Ollama instances
pkill -f "ollama serve" || true
sleep 2

# Start Ollama instance for each GPU
for i in {0..4}; do
    PORT=$((BASE_PORT + i))
    echo "Starting Ollama on GPU $i, port $PORT..."

    CUDA_VISIBLE_DEVICES=$i nohup ollama serve \
        --host 0.0.0.0 \
        --port $PORT \
        > "ollama_gpu${i}.log" 2>&1 &

    echo "  GPU $i Ollama PID: $!"
done

# Wait for Ollama instances to start
echo "Waiting for Ollama instances to initialize..."
sleep 10

# Start OLOL servers for each Ollama instance
for i in {0..4}; do
    OLLAMA_PORT=$((BASE_PORT + i))
    OLOL_PORT=$((OLOL_BASE_PORT + i))

    echo "Starting OLOL server for GPU $i..."
    nohup olol server \
        --host 0.0.0.0 \
        --port $OLOL_PORT \
        --ollama-host "http://localhost:$OLLAMA_PORT" \
        > "olol_gpu${i}.log" 2>&1 &

    echo "  GPU $i OLOL PID: $!"
done

echo ""
echo "Setup complete!"
echo "Ollama instances running on ports: 11434-11438"
echo "OLOL servers running on ports: 50051-50055"
echo ""
echo "Test with:"
echo "  curl http://localhost:11434/api/tags"
echo "  curl http://localhost:11435/api/tags"
```

Make it executable:
```bash
chmod +x start_ollama_5gpu.sh
./start_ollama_5gpu.sh
```

### Configuration 2: 3× Mixed GPUs (e.g., RTX 3060 + A2000 + A4000)

Create `start_ollama_3gpu.sh`:

```bash
#!/bin/bash
# Startup script for 3× mixed GPU Ollama instances

set -e

echo "Starting 3× Ollama instances for mixed GPUs..."

# Configuration
GPU_NAMES=("RTX_3060" "A2000_12GB" "A4000")
BASE_PORT=11434
OLOL_BASE_PORT=50051

# Kill any existing Ollama instances
pkill -f "ollama serve" || true
sleep 2

# Start Ollama instance for each GPU
for i in {0..2}; do
    PORT=$((BASE_PORT + i))
    GPU_NAME=${GPU_NAMES[$i]}

    echo "Starting Ollama on GPU $i (${GPU_NAME}), port $PORT..."

    CUDA_VISIBLE_DEVICES=$i nohup ollama serve \
        --host 0.0.0.0 \
        --port $PORT \
        > "ollama_${GPU_NAME}.log" 2>&1 &

    echo "  GPU $i (${GPU_NAME}) Ollama PID: $!"
done

# Wait for Ollama instances to start
echo "Waiting for Ollama instances to initialize..."
sleep 10

# Start OLOL servers for each Ollama instance
for i in {0..2}; do
    OLLAMA_PORT=$((BASE_PORT + i))
    OLOL_PORT=$((OLOL_BASE_PORT + i))
    GPU_NAME=${GPU_NAMES[$i]}

    echo "Starting OLOL server for GPU $i (${GPU_NAME})..."
    nohup olol server \
        --host 0.0.0.0 \
        --port $OLOL_PORT \
        --ollama-host "http://localhost:$OLLAMA_PORT" \
        > "olol_${GPU_NAME}.log" 2>&1 &

    echo "  GPU $i (${GPU_NAME}) OLOL PID: $!"
done

echo ""
echo "Setup complete!"
echo "Ollama instances:"
echo "  GPU 0 (RTX_3060):  http://localhost:11434"
echo "  GPU 1 (A2000_12GB): http://localhost:11435"
echo "  GPU 2 (A4000):      http://localhost:11436"
echo ""
echo "OLOL servers: 50051-50053"
```

### Unified Startup Script (Configurable)

Create `start_ollama_multi.sh`:

```bash
#!/bin/bash
# Universal multi-GPU Ollama startup script

set -e

# Configuration - EDIT THIS
NUM_GPUS=5  # Change to your number of GPUs
BASE_PORT=11434
OLOL_BASE_PORT=50051

echo "Starting $NUM_GPUS Ollama instances..."

# Kill any existing instances
pkill -f "ollama serve" || true
pkill -f "olol server" || true
sleep 2

# Start Ollama for each GPU
for ((i=0; i<NUM_GPUS; i++)); do
    PORT=$((BASE_PORT + i))

    echo "Starting Ollama on GPU $i, port $PORT..."
    CUDA_VISIBLE_DEVICES=$i nohup ollama serve \
        --host 0.0.0.0 \
        --port $PORT \
        > "ollama_gpu${i}.log" 2>&1 &
done

sleep 10

# Start OLOL servers
for ((i=0; i<NUM_GPUS; i++)); do
    OLLAMA_PORT=$((BASE_PORT + i))
    OLOL_PORT=$((OLOL_BASE_PORT + i))

    echo "Starting OLOL server for GPU $i..."
    nohup olol server \
        --host 0.0.0.0 \
        --port $OLOL_PORT \
        --ollama-host "http://localhost:$OLLAMA_PORT" \
        > "olol_gpu${i}.log" 2>&1 &
done

echo "Setup complete! Running $NUM_GPUS instances."
```

## Systemd Service (Production Setup)

For automatic startup on boot, create systemd services:

### Create Service Template

Create `/etc/systemd/system/ollama@.service`:

```ini
[Unit]
Description=Ollama Service for GPU %i
After=network.target

[Service]
Type=simple
User=your-username
Environment="CUDA_VISIBLE_DEVICES=%i"
ExecStart=/usr/local/bin/ollama serve --host 0.0.0.0 --port 1143%i
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Enable Services

```bash
# Enable for GPUs 0-4
sudo systemctl enable ollama@0
sudo systemctl enable ollama@1
sudo systemctl enable ollama@2
sudo systemctl enable ollama@3
sudo systemctl enable ollama@4

# Start all
sudo systemctl start ollama@{0..4}

# Check status
sudo systemctl status ollama@*
```

## Management and Monitoring

### Check All Running Instances

```bash
# List all Ollama processes
ps aux | grep "ollama serve"

# Check ports
netstat -tlnp | grep ollama

# Or use ss
ss -tlnp | grep ollama
```

### Stop All Instances

```bash
#!/bin/bash
# stop_ollama.sh

echo "Stopping all Ollama and OLOL instances..."
pkill -f "ollama serve"
pkill -f "olol server"
echo "All instances stopped."
```

### Restart Single Instance

```bash
# Stop GPU 2 instance
kill $(ps aux | grep "CUDA_VISIBLE_DEVICES=2.*ollama serve" | grep -v grep | awk '{print $2}')

# Restart GPU 2
CUDA_VISIBLE_DEVICES=2 nohup ollama serve --host 0.0.0.0 --port 11436 > ollama_gpu2.log 2>&1 &
```

### Monitor GPU Utilization

```bash
# Real-time monitoring of all GPUs
watch -n 1 nvidia-smi

# Or with more detail
nvidia-smi dmon -s pucvmet -d 1
```

### Check Logs

```bash
# View logs for specific GPU
tail -f ollama_gpu0.log
tail -f olol_gpu0.log

# View all logs
tail -f ollama_gpu*.log olol_gpu*.log
```

## Loading Models on Each Instance

### Pre-load Models on All GPUs

```bash
#!/bin/bash
# load_models.sh - Load models across all Ollama instances

BASE_PORT=11434
MODELS=("mistral:7b" "llama3:8b" "qwen2:7b")

for i in {0..4}; do
    PORT=$((BASE_PORT + i))
    MODEL=${MODELS[$i % ${#MODELS[@]}]}  # Cycle through models

    echo "Loading $MODEL on GPU $i (port $PORT)..."
    curl -X POST "http://localhost:$PORT/api/pull" \
        -d "{\"name\":\"$MODEL\"}" &
done

wait
echo "All models loaded!"
```

### Strategic Model Loading

```bash
# Load different models based on GPU capabilities
# GPU 0-4 (8GB each): 7B models
curl http://localhost:11434/api/pull -d '{"name":"mistral:7b-q8_0"}'
curl http://localhost:11435/api/pull -d '{"name":"llama3:8b-q8_0"}'

# GPU 5 (12GB): Larger model
curl http://localhost:11439/api/pull -d '{"name":"qwen2.5:14b-q8_0"}'
```

## Verification and Testing

### Test Each Instance

```bash
#!/bin/bash
# test_instances.sh

BASE_PORT=11434

for i in {0..4}; do
    PORT=$((BASE_PORT + i))
    echo "Testing GPU $i (port $PORT)..."

    curl -s "http://localhost:$PORT/api/tags" | jq '.models[].name' || echo "Failed"
done
```

### Test Inference on Each GPU

```bash
# Quick inference test
for port in 11434 11435 11436 11437 11438; do
    echo "Testing port $port..."
    curl -X POST "http://localhost:$port/api/generate" \
        -d '{"model":"mistral:7b","prompt":"Hello!","stream":false}' | jq .response
done
```

## Common Issues and Solutions

### Issue: Port Already in Use

**Error**: `bind: address already in use`

**Solution**:
```bash
# Find what's using the port
lsof -i :11434

# Kill the process
kill -9 <PID>
```

### Issue: GPU Not Found

**Error**: `CUDA error: no CUDA-capable device is detected`

**Solution**:
```bash
# Verify GPUs are visible
nvidia-smi

# Check CUDA installation
nvcc --version

# Ensure drivers are loaded
nvidia-smi
```

### Issue: Out of Memory

**Error**: `CUDA out of memory`

**Solution**:
- Use smaller models
- Increase quantization (q8→q5→q4)
- Ensure only one model per GPU
- Check for leftover processes

### Issue: Instances Not Starting

**Solution**:
```bash
# Check logs
tail -f ollama_gpu*.log

# Verify ports are free
netstat -tlnp | grep 1143

# Check CUDA devices
echo $CUDA_VISIBLE_DEVICES
```

## Performance Tuning

### Optimize for Throughput

```bash
# Disable logging for performance
CUDA_VISIBLE_DEVICES=0 ollama serve --host 0.0.0.0 --port 11434 2>/dev/null &
```

### Set Resource Limits

```bash
# Limit context window to save VRAM
curl -X POST http://localhost:11434/api/generate \
    -d '{"model":"mistral:7b","prompt":"test","options":{"num_ctx":2048}}'
```

## Summary

**Key Principles**:
1. One Ollama instance per GPU for maximum utilization
2. Use `CUDA_VISIBLE_DEVICES` to bind instances to specific GPUs
3. Use different ports for each instance
4. Use OLOL to coordinate requests across all instances

**Benefits**:
- All GPUs stay busy processing different requests
- Maximum throughput for batch processing
- Flexibility to load different models on different GPUs
- Easy scaling by adding more instances
