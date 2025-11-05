# GNU Parallel for Ollama Batch Processing

A practical guide to using GNU parallel for batch processing inference requests with Ollama through the command line.

## Why GNU Parallel for Ollama

GNU parallel provides a simple shell-based approach to batch processing that:
- Doesn't require Python or other programming languages
- Works directly with curl and Ollama's API
- Provides built-in job management and concurrency control
- Integrates easily with Unix tools for monitoring

## Basic Setup

### Prerequisites

```bash
# Install GNU parallel
# Ubuntu/Debian
sudo apt install parallel

# macOS
brew install parallel

# CentOS/RHEL
sudo yum install parallel

# Verify Ollama is running
curl -s http://localhost:11434/api/tags
```

### Create Prompt Files

Each prompt should be a separate JSON file:

```bash
# Create prompt files
echo '{"model": "llama2", "prompt": "Explain AI", "stream": false}' > prompt1.json
echo '{"model": "llama2", "prompt": "Write a poem", "stream": false}' > prompt2.json
echo '{"model": "llama2", "prompt": "Summarize climate change", "stream": false}' > prompt3.json
```

**Important**: Always include `"stream": false` to get complete responses.

## Basic Parallel Command

```bash
parallel -j 3 'curl -s -X POST http://localhost:11434/api/generate -d @{} > result_{#}.json' ::: prompt*.json
```

### Command Breakdown

- `parallel` - GNU parallel command
- `-j 3` - Run maximum 3 jobs concurrently
- `'curl ...'` - Command template for each job
  - `-s` - Silent mode (no progress bars)
  - `-X POST` - HTTP POST method
  - `http://localhost:11434/api/generate` - Ollama API endpoint
  - `-d @{}` - Send data from file (`{}` = current input filename)
  - `> result_{#}.json` - Save output (`{#}` = job number)
- `:::` - Input separator
- `prompt*.json` - Input files (wildcard pattern)

### What Parallel Does

For files `prompt1.json`, `prompt2.json`, `prompt3.json`, parallel essentially runs:

```bash
curl -s -X POST http://localhost:11434/api/generate -d @prompt1.json > result_1.json
curl -s -X POST http://localhost:11434/api/generate -d @prompt2.json > result_2.json
curl -s -X POST http://localhost:11434/api/generate -d @prompt3.json > result_3.json
```

But it manages concurrency, handles errors, and tracks progress automatically.

## Running in Background

### Send to Background

```bash
parallel -j 3 'curl -s -X POST http://localhost:11434/api/generate -d @{} > result_{#}.json' ::: prompt*.json &
```

The `&` at the end sends the entire job to background, returning control to your terminal.

### Background Job Management

```bash
# List running background jobs
jobs
jobs -l  # With process IDs

# Bring job to foreground
fg       # Most recent job
fg %1    # Specific job number

# Send foreground job to background
# Press: Ctrl+Z (suspends job)
# Then type: bg (resumes in background)

# Kill a background job
kill %1      # By job number
kill [PID]   # By process ID
```

### Using nohup for Persistence

For long-running jobs that should continue even if you log out:

```bash
nohup parallel -j 3 'curl -s -X POST http://localhost:11434/api/generate -d @{} > result_{#}.json' ::: prompt*.json &
```

## Monitoring Progress

### Using watch Command

Monitor file creation and growth:

```bash
# Basic monitoring
watch -n 2 'wc -w result_*.json 2>/dev/null'

# With change highlighting
watch -n 2 --differences 'wc -w result_*.json 2>/dev/null'

# With both word and character counts
watch -n 2 --differences 'wc -wc result_*.json 2>/dev/null'

# With headers for clarity
watch -n 2 --differences 'echo "     WORDS FILENAME"; echo "     ----- --------"; wc -w result_*.json 2>/dev/null'
```

### Watch Command Breakdown

- `watch` - Repeatedly execute command and display results
- `-n 2` - Refresh every 2 seconds
- `--differences` - Highlight changes between refreshes
- `'wc -wc result_*.json'` - Count words and characters
- `2>/dev/null` - Suppress "file not found" errors

### Alternative Monitoring Methods

Simple refresh loop (better for small terminals):

```bash
while true; do
    clear
    echo "=== Progress ($(date)) ==="
    wc -w result_*.json 2>/dev/null
    echo ""
    echo "Completed: $(ls result_*.json 2>/dev/null | wc -l) files"
    sleep 2
done
# Stop with Ctrl+C
```

Check completion ratio:

```bash
watch -n 2 'echo "Progress: $(ls result_*.json 2>/dev/null | wc -l)/$(ls prompt_*.json 2>/dev/null | wc -l)"'
```

## Timeout and Error Handling

### Adding Timeouts

Ollama responses can take time. Add appropriate timeouts:

```bash
# With 5-minute timeout per request
parallel -j 3 'curl -s --max-time 300 -X POST http://localhost:11434/api/generate -d @{} > result_{#}.json' ::: prompt*.json &

# With connection timeout and retries
parallel -j 3 'curl -s --connect-timeout 10 --max-time 600 --retry 2 -X POST http://localhost:11434/api/generate -d @{} > result_{#}.json' ::: prompt*.json &
```

### Timeout Options

- `--connect-timeout 10` - Max 10 seconds to establish connection
- `--max-time 300` - Max 300 seconds (5 minutes) for complete operation
- `--retry 2` - Retry failed requests up to 2 times

### Using timeout Command

Wrap entire parallel job with timeout:

```bash
timeout 300 parallel -j 3 'curl -s -X POST http://localhost:11434/api/generate -d @{} > result_{#}.json' ::: prompt*.json
```

This kills the entire job after 300 seconds if not completed.

## Adjusting Concurrency

### Finding Optimal Concurrency

Start conservative and increase:

```bash
# Start with sequential processing
parallel -j 1 'curl -s --max-time 300 -X POST http://localhost:11434/api/generate -d @{} > result_{#}.json' ::: prompt*.json

# If that works, try 2-3 concurrent jobs
parallel -j 2 ...
parallel -j 3 ...
```

### Important: Ollama's Sequential Nature

**Key Understanding**: Even with multiple concurrent curl requests, Ollama typically processes inference **sequentially** (one at a time). This is by design:

- **GPU Memory**: Most setups can't fit multiple model instances in VRAM
- **Resource Management**: Ollama queues requests internally
- **Optimization**: Ensures maximum VRAM available for each inference

**What parallel DOES help with**:
- Requests are queued immediately (no gaps between processing)
- Network I/O happens concurrently
- File writing is concurrent
- No manual waiting between jobs

For true parallel inference, you need:
- Multiple Ollama instances on different ports
- Multiple GPUs with separate instances
- Distributed setup like OLOL

### Use Cases for Different Concurrency Levels

```bash
# Single job (-j 1): Testing, debugging, limited resources
parallel -j 1 ...

# Low concurrency (-j 2-3): Standard use, ensures queue stays full
parallel -j 3 ...

# High concurrency (-j 10+): With OLOL distributed setup
parallel -j 10 ...
```

## Troubleshooting

### Empty Result Files

**Symptom**: Files created but have 0 bytes or no content

**Check**:
```bash
# Verify Ollama is accessible
curl -s http://localhost:11434/api/tags

# Test single request manually
curl -v -X POST http://localhost:11434/api/generate -d @prompt1.json > test.json 2> debug.log
cat test.json
cat debug.log

# Verify prompt file format
cat prompt1.json
```

**Common causes**:
- Missing `"stream": false` in prompt JSON
- Wrong model name (check with `ollama list`)
- Malformed JSON in prompt files
- Timeout too short for long responses

### "Client Closing Connection" Errors

**Symptom**: In `ollama serve` logs: "aborting completion request due to client closing the connection"

**Solution**: Increase curl timeout:
```bash
parallel -j 1 'curl -s --max-time 300 -X POST http://localhost:11434/api/generate -d @{} > result_{#}.json' ::: prompt*.json
```

**Reduce concurrency** if multiple jobs overwhelm Ollama:
```bash
# Try with just 1 concurrent job
parallel -j 1 ...
```

### Stuck Parallel Command

**Symptom**: Parallel won't respond to Ctrl+C

**Solutions**:
```bash
# Find and kill the parallel process
ps aux | grep parallel
kill [PID]
kill -9 [PID]  # Force kill if needed

# Or kill all parallel processes
pkill parallel
pkill -9 parallel  # Force kill

# Also kill curl processes
pkill -f "curl.*11434"

# Nuclear option
killall parallel curl
```

### Terminal Corruption

**Symptom**: After backgrounding `ollama run`, terminal shows garbled text and constant redrawing

**Why**: `ollama run` is interactive and tries to control the terminal even when backgrounded

**Immediate fix**:
```bash
reset      # Reset terminal
clear      # Clear screen
stty sane  # Reset terminal settings
```

**Prevention**: Don't background `ollama run`. Use the API instead:
```bash
# Wrong approach
ollama run llama2 &  # Causes terminal issues

# Right approach
ollama serve &  # Run server in background
# Then use curl/parallel with API
```

### Syntax Errors with Parallel

**Symptom**: Cursor shows `>` and command won't execute

**Cause**: Unclosed quote or incomplete command

**Fix**:
```bash
# Press Ctrl+C to cancel
# Then check for:
# - Matching quotes (' and ')
# - Proper ::: separator
# - Complete command syntax
```

## Complete Working Example

```bash
#!/bin/bash
# batch_ollama.sh - Complete batch processing script

# Configuration
MODEL="llama2"
NUM_JOBS=3
TIMEOUT=300

# Create prompt files
echo "Creating prompt files..."
for i in {1..10}; do
    echo "{\"model\": \"$MODEL\", \"prompt\": \"Write a haiku about number $i\", \"stream\": false}" > prompt_$i.json
done

# Run batch processing in background
echo "Starting batch processing with $NUM_JOBS concurrent jobs..."
parallel -j $NUM_JOBS \
    "curl -s --max-time $TIMEOUT -X POST http://localhost:11434/api/generate -d @{} > result_{#}.json" \
    ::: prompt_*.json &

PARALLEL_PID=$!
echo "Batch job started (PID: $PARALLEL_PID)"

# Monitor in foreground
echo "Monitoring progress (Ctrl+C to stop monitoring)..."
while kill -0 $PARALLEL_PID 2>/dev/null; do
    clear
    echo "=== Batch Processing Progress ==="
    echo "Time: $(date)"
    echo ""
    TOTAL=$(ls prompt_*.json 2>/dev/null | wc -l)
    COMPLETED=$(ls result_*.json 2>/dev/null | wc -l)
    echo "Completed: $COMPLETED / $TOTAL"
    echo ""
    echo "Recent results:"
    ls -lt result_*.json 2>/dev/null | head -5
    sleep 2
done

echo ""
echo "Batch processing complete!"
echo "Results saved in result_*.json files"
```

Usage:
```bash
chmod +x batch_ollama.sh
./batch_ollama.sh
```

## Best Practices

1. **Always set timeouts**: Prevent hanging requests
2. **Start with -j 1**: Verify setup works before increasing concurrency
3. **Include "stream": false**: Ensure complete responses
4. **Monitor actively**: Use watch or custom monitoring loops
5. **Test manually first**: Run one curl command to verify format
6. **Use background jobs for long batches**: Free up terminal with `&`
7. **Redirect errors**: Use `2>/dev/null` to suppress noise
8. **Check Ollama logs**: Monitor `ollama serve` output for issues

## Performance Tips

- **Optimal concurrency**: Usually 1-3 jobs for standard Ollama setup
- **Larger batches**: Use background jobs and monitoring
- **Long prompts**: Increase `--max-time` significantly
- **Resource monitoring**: Watch GPU utilization with `nvidia-smi`
- **File organization**: Use descriptive filenames for easier debugging

## Summary

GNU parallel with curl provides a lightweight, flexible way to batch process Ollama inference requests:
- No programming required
- Built-in concurrency management
- Easy monitoring with watch
- Integrates with Unix tools
- Perfect for ad-hoc batch jobs

For production systems with complex workflows, consider Python-based approaches (asyncio, Celery) or distributed setups (OLOL).
