# LingBot World Fast Performance Benchmark

This benchmark measures TeleFuser's steady-state LingBot World Fast chunk
generation latency, throughput, and CUDA memory usage through the pipeline API.

## Default configuration

- 201 output frames, chunk size 3, BF16
- local attention size 18 and sink size 6
- DiT, VAE, and text encoder on `cuda:0`
- one warmup chunk excluded from steady-state statistics
- MP4 encoding disabled

The bundled case 03 camera-control input comes from the official LingBot World
repository. See [`data/case03/SOURCE.md`](data/case03/SOURCE.md) for provenance.

## Run

Set the model-zoo root and run from the TeleFuser repository root:

```bash
export TF_MODEL_ZOO_PATH=/path/to/model_zoo

python benchmarks/lingbot_world_fast/benchmark.py
```

The default model paths are:

```text
$TF_MODEL_ZOO_PATH/Wan2.2-I2V-A14B
$TF_MODEL_ZOO_PATH/lingbot-world-fast
```

Override them independently when the model directory layout differs:

```bash
python benchmarks/lingbot_world_fast/benchmark.py \
  --checkpoint-dir /path/to/Wan2.2-I2V-A14B \
  --fast-checkpoint-dir /path/to/lingbot-world-fast
```

The default report is written to
`work_dirs/benchmarks/lingbot_world_fast/<timestamp>/result.json`. Use
`--output-dir` for a fixed location. To write a diagnostic video without
including encoding in the chunk compute summary, pass `--save-video`.

Run `python benchmarks/lingbot_world_fast/benchmark.py --help` for all runtime,
device, input, and output options. Each invocation measures one configuration;
run the command separately for full/local-attention comparisons. Full attention
uses `--local-attn-size -1 --sink-size 0`.

## Report

Schema version 1 records:

- the requested configuration and actual runtime/KV geometry;
- TeleFuser commit, Python, PyTorch, CUDA, and GPU information;
- pipeline initialization, runtime creation, and generation elapsed time;
- phase-specific peak allocated and reserved CUDA memory;
- every chunk's synchronized compute time;
- steady-state mean, p50, p90, minimum, maximum, standard deviation, and frames/s;
- optional video encoding time and artifact path.

The benchmark only reports measurements. It does not apply hardware-specific
pass/fail thresholds or compare against a baseline automatically.

## Torch profiler diagnostics

Profiler traces perturb timing and should not be compared with the normal
benchmark report. Use TeleFuser's existing profiler ranges only for diagnosis:

```bash
export TELEFUSER_PROFILE_DEBUG=true
export TELEFUSER_PIPELINE_NAME=lingbot_world_fast
export TELEFUSER_PROFILER_OUTPUT_DIR=/path/to/profile_output
export ENABLE_PROFILER_NAMES=generate_next_chunk,denoise_chunk,kv_cache_update_forward,vae_decode

python benchmarks/lingbot_world_fast/benchmark.py --output-dir work_dirs/lingbot_profile_run
```

Useful ranges are `generate_next_chunk`, `denoise_chunk`,
`kv_cache_update_forward`, and `vae_decode`.
