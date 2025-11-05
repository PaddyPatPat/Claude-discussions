# LLM Batch Processing Strategies

Comprehensive guide for processing large batches of inference requests across distributed Ollama instances.

## What is Batch Processing

**Batch processing** = processing many inference requests as a queue, maximizing hardware utilization by keeping all GPUs constantly busy.

### Key Goals

1. **Maximum GPU Utilization**: Keep all GPUs at ~100% utilization
2. **High Throughput**: Process as many requests per minute as possible
3. **Reliability**: Handle failures gracefully and retry as needed
4. **Monitoring**: Track progress and identify bottlenecks

## Queue Management Strategies

### Strategy 1: Simple Python ThreadPool

Best for: Small to medium batches (<10,000 requests), simple setups

```python
import requests
from concurrent.futures import ThreadPoolExecutor
import json

# Configuration
OLOL_ENDPOINT = "http://localhost:8000/api/generate"
MAX_WORKERS = 20  # Concurrent requests

# Your batch of prompts
prompts = [
    "Explain quantum computing",
    "Write a story about AI",
    # ... thousands more
]

def process_request(prompt, model="mistral:7b"):
    """Process a single inference request."""
    try:
        response = requests.post(
            OLOL_ENDPOINT,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=120  # 2 minute timeout
        )
        response.raise_for_status()
        return {
            "prompt": prompt,
            "response": response.json()["response"],
            "status": "success"
        }
    except Exception as e:
        return {
            "prompt": prompt,
            "error": str(e),
            "status": "failed"
        }

# Process in parallel
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    results = list(executor.map(lambda p: process_request(p), prompts))

# Save results
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

# Print statistics
successes = sum(1 for r in results if r["status"] == "success")
failures = sum(1 for r in results if r["status"] == "failed")
print(f"Completed: {successes}, Failed: {failures}")
```

**Pros**:
- Simple to implement
- Built-in Python, no dependencies
- Good for straightforward batches

**Cons**:
- No automatic retry
- Limited monitoring
- Memory issues with huge batches

### Strategy 2: Asyncio for Higher Concurrency

Best for: Large batches (10,000+), better performance

```python
import asyncio
import aiohttp
import json
from tqdm import tqdm

OLOL_ENDPOINT = "http://localhost:8000/api/generate"
MAX_CONCURRENT = 50  # Higher concurrency with async

async def process_request(session, prompt, model="mistral:7b", semaphore=None):
    """Process single request with rate limiting."""
    async with semaphore:
        try:
            async with session.post(
                OLOL_ENDPOINT,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                result = await response.json()
                return {
                    "prompt": prompt,
                    "response": result["response"],
                    "status": "success"
                }
        except Exception as e:
            return {
                "prompt": prompt,
                "error": str(e),
                "status": "failed"
            }

async def process_batch(prompts, max_concurrent=50):
    """Process entire batch with progress bar."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async with aiohttp.ClientSession() as session:
        tasks = [
            process_request(session, prompt, semaphore=semaphore)
            for prompt in prompts
        ]

        results = []
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            result = await task
            results.append(result)

        return results

# Run
prompts = ["..." for _ in range(10000)]  # Your prompts
results = asyncio.run(process_batch(prompts, max_concurrent=50))
```

**Pros**:
- Much higher concurrency
- Better performance
- Progress tracking with tqdm

**Cons**:
- More complex code
- Requires async understanding

### Strategy 3: Celery for Production

Best for: Production systems, distributed workers, fault tolerance

```python
# tasks.py
from celery import Celery
import requests

app = Celery('llm_tasks', broker='redis://localhost:6379/0')

@app.task(bind=True, max_retries=3)
def process_inference(self, prompt, model="mistral:7b"):
    """Celery task for single inference with retry."""
    try:
        response = requests.post(
            "http://localhost:8000/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as exc:
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)

# main.py
from tasks import process_inference
from celery import group

prompts = ["..." for _ in range(10000)]

# Create task group
job = group(process_inference.s(prompt) for prompt in prompts)
result = job.apply_async()

# Wait for completion
results = result.get()
```

**Pros**:
- Production-grade
- Automatic retry
- Distributed workers
- Fault tolerance
- Monitoring with Flower

**Cons**:
- Requires Redis/RabbitMQ
- More infrastructure
- Steeper learning curve

## Optimizing for Maximum Throughput

### Tuning Concurrency

The key is finding the sweet spot for `max_workers` or `MAX_CONCURRENT`:

```python
# Too low: GPUs idle
MAX_WORKERS = 5  # With 12 GPU instances = underutilized

# Too high: Timeouts and failures
MAX_WORKERS = 500  # Overwhelms servers

# Just right: Matches compute capacity
MAX_WORKERS = 20-50  # For 12 compute instances
```

**Formula**:
```
Optimal concurrency ≈ (Number of compute instances × 2-4)

Example:
12 instances × 3 = 36 concurrent requests
```

### Progress Tracking

```python
from tqdm import tqdm
import time

def process_with_progress(prompts, max_workers=20):
    """Process batch with detailed progress tracking."""
    results = []
    failed = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_request, p): p
            for p in prompts
        }

        with tqdm(total=len(prompts)) as pbar:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result["status"] == "success":
                    results.append(result)
                else:
                    failed.append(result)

                pbar.update(1)
                pbar.set_postfix({
                    "success": len(results),
                    "failed": len(failed),
                    "rate": f"{len(results)/(time.time() - start_time):.2f}/s"
                })

    return results, failed
```

### Retry Logic

```python
def process_with_retry(prompt, model="mistral:7b", max_retries=3):
    """Process with automatic retry on failure."""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                OLOL_ENDPOINT,
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=120
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            if attempt == max_retries - 1:
                # Final attempt failed
                raise
            # Wait before retry (exponential backoff)
            time.sleep(2 ** attempt)
```

## Model Selection Strategies

### Strategy 1: Single Model (Simplest)

```python
# All requests use same model
results = process_batch(prompts, model="mistral:7b")
```

**Pros**: Simple, predictable
**Cons**: Doesn't leverage different model strengths

### Strategy 2: Different Models for Different Tasks

```python
# Route to appropriate model based on task
def select_model(prompt):
    if "code" in prompt.lower():
        return "codellama:34b"
    elif len(prompt) > 1000:
        return "llama3:70b"
    else:
        return "mistral:7b"

results = [
    process_request(p, model=select_model(p))
    for p in prompts
]
```

**Pros**: Optimized for task type
**Cons**: More complex routing logic

### Strategy 3: Load Balancing Across Models

```python
# Distribute across multiple models for throughput
models = ["mistral:7b", "llama3:8b", "qwen2:7b"]

results = [
    process_request(p, model=models[i % len(models)])
    for i, p in enumerate(prompts)
]
```

**Pros**: Better GPU utilization
**Cons**: Mixed quality/characteristics

## Monitoring GPU Utilization During Batch

### Real-Time GPU Monitoring

```bash
# Terminal 1: Run batch processing
python batch_process.py

# Terminal 2: Monitor GPUs
watch -n 1 nvidia-smi
```

### Automated Monitoring

```python
import subprocess
import threading
import time

def monitor_gpus(interval=5, duration=3600):
    """Monitor GPU utilization during batch processing."""
    start_time = time.time()
    stats = []

    while time.time() - start_time < duration:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True
        )

        timestamp = time.time() - start_time
        gpu_stats = result.stdout.strip().split('\n')

        stats.append({
            "timestamp": timestamp,
            "gpus": [
                {"util": int(s.split(',')[0]), "mem": int(s.split(',')[1])}
                for s in gpu_stats
            ]
        })

        time.sleep(interval)

    return stats

# Run monitoring in background
monitor_thread = threading.Thread(
    target=lambda: monitor_gpus(interval=5, duration=3600),
    daemon=True
)
monitor_thread.start()

# Run your batch processing
results = process_batch(prompts)
```

## Handling Different Prompt Lengths

### Challenge: Variable Processing Time

Short prompts (10 tokens) vs long prompts (1000 tokens) = vastly different processing times.

### Solution 1: Batch by Length

```python
# Group prompts by similar length
short_prompts = [p for p in prompts if len(p) < 100]
medium_prompts = [p for p in prompts if 100 <= len(p) < 500]
long_prompts = [p for p in prompts if len(p) >= 500]

# Process each group separately
results_short = process_batch(short_prompts, max_workers=50)
results_medium = process_batch(medium_prompts, max_workers=30)
results_long = process_batch(long_prompts, max_workers=15)
```

### Solution 2: Dynamic Timeout

```python
def process_with_dynamic_timeout(prompt, model="mistral:7b"):
    """Adjust timeout based on prompt length."""
    # Base timeout + extra for longer prompts
    timeout = 30 + (len(prompt) // 100)  # +1 second per 100 chars

    response = requests.post(
        OLOL_ENDPOINT,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout
    )
    return response.json()
```

## Saving and Checkpointing

### Incremental Saves

```python
import pickle
import os

CHECKPOINT_FILE = "batch_checkpoint.pkl"
RESULTS_FILE = "batch_results.json"

def process_with_checkpointing(prompts, checkpoint_interval=100):
    """Save progress periodically to allow resume."""

    # Load checkpoint if exists
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "rb") as f:
            completed_indices = pickle.load(f)
    else:
        completed_indices = set()

    results = []

    for i, prompt in enumerate(prompts):
        if i in completed_indices:
            continue  # Skip already completed

        result = process_request(prompt)
        results.append(result)
        completed_indices.add(i)

        # Save checkpoint periodically
        if len(completed_indices) % checkpoint_interval == 0:
            with open(CHECKPOINT_FILE, "wb") as f:
                pickle.dump(completed_indices, f)
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f)

    # Final save
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f)
    os.remove(CHECKPOINT_FILE)

    return results
```

## Performance Benchmarking

### Measure Throughput

```python
import time

def benchmark_batch(prompts, max_workers=20):
    """Benchmark batch processing performance."""
    start_time = time.time()

    results = process_batch(prompts, max_workers=max_workers)

    end_time = time.time()
    duration = end_time - start_time

    stats = {
        "total_prompts": len(prompts),
        "duration_seconds": duration,
        "throughput_per_second": len(prompts) / duration,
        "avg_seconds_per_request": duration / len(prompts),
        "max_workers": max_workers
    }

    print(f"Processed {len(prompts)} requests in {duration:.2f}s")
    print(f"Throughput: {stats['throughput_per_second']:.2f} requests/sec")

    return results, stats
```

### Compare Different Configurations

```python
# Test different concurrency levels
for workers in [10, 20, 30, 50]:
    results, stats = benchmark_batch(test_prompts, max_workers=workers)
    print(f"Workers: {workers}, Throughput: {stats['throughput_per_second']:.2f}/s")
```

## Common Pitfalls

### Pitfall 1: Too Much Concurrency

**Problem**: Timeouts, failed requests
**Solution**: Reduce `max_workers` to match compute capacity

### Pitfall 2: Not Enough Concurrency

**Problem**: GPUs idle, low throughput
**Solution**: Increase `max_workers` until GPU utilization hits 95%+

### Pitfall 3: Memory Leaks

**Problem**: Memory usage grows over time
**Solution**: Process in smaller chunks, clear results periodically

```python
# Process in chunks
CHUNK_SIZE = 1000

for i in range(0, len(prompts), CHUNK_SIZE):
    chunk = prompts[i:i+CHUNK_SIZE]
    results = process_batch(chunk)
    save_results(results)
    del results  # Free memory
```

### Pitfall 4: No Error Handling

**Problem**: One failure stops entire batch
**Solution**: Catch exceptions, continue processing, log failures

## Best Practices Summary

1. **Start conservative**: Begin with low concurrency, increase until GPU utilization plateaus
2. **Monitor continuously**: Watch GPU utilization, adjust concurrency
3. **Checkpoint progress**: Save results periodically
4. **Handle failures**: Implement retry logic
5. **Batch strategically**: Group similar prompts
6. **Measure performance**: Benchmark different configurations
7. **Use appropriate tools**: ThreadPool for simple, async for large, Celery for production

## Example: Complete Production Script

```python
#!/usr/bin/env python3
"""
Production-ready batch LLM inference script.
"""
import asyncio
import aiohttp
import json
import argparse
from tqdm import tqdm
from pathlib import Path

async def process_request(session, prompt, model, semaphore, retries=3):
    """Process single request with retry."""
    async with semaphore:
        for attempt in range(retries):
            try:
                async with session.post(
                    "http://localhost:8000/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False},
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    result = await response.json()
                    return {"prompt": prompt, "response": result["response"], "status": "success"}
            except Exception as e:
                if attempt == retries - 1:
                    return {"prompt": prompt, "error": str(e), "status": "failed"}
                await asyncio.sleep(2 ** attempt)

async def main(input_file, output_file, model, max_concurrent):
    """Main batch processing function."""
    # Load prompts
    with open(input_file) as f:
        prompts = [line.strip() for line in f if line.strip()]

    print(f"Processing {len(prompts)} prompts with model {model}")
    print(f"Max concurrent: {max_concurrent}")

    semaphore = asyncio.Semaphore(max_concurrent)
    results = []

    async with aiohttp.ClientSession() as session:
        tasks = [process_request(session, p, model, semaphore) for p in prompts]

        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            result = await task
            results.append(result)

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Statistics
    successes = sum(1 for r in results if r["status"] == "success")
    failures = sum(1 for r in results if r["status"] == "failed")

    print(f"\nCompleted: {successes}, Failed: {failures}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch LLM inference")
    parser.add_argument("input", help="Input file with prompts (one per line)")
    parser.add_argument("-o", "--output", default="results.json", help="Output JSON file")
    parser.add_argument("-m", "--model", default="mistral:7b", help="Model to use")
    parser.add_argument("-c", "--concurrent", type=int, default=20, help="Max concurrent requests")

    args = parser.parse_args()
    asyncio.run(main(args.input, args.output, args.model, args.concurrent))
```

Usage:
```bash
python batch_inference.py prompts.txt -o results.json -m mistral:7b -c 30
```
