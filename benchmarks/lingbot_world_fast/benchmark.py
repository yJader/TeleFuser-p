from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CASE_DIR = SCRIPT_DIR / "data" / "case03"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "work_dirs" / "benchmarks" / "lingbot_world_fast"


@dataclass(frozen=True)
class BenchmarkConfig:
    checkpoint_dir: Path
    fast_checkpoint_dir: Path
    case_dir: Path = DEFAULT_CASE_DIR
    output_dir: Path | None = None
    frame_num: int = 201
    chunk_size: int = 3
    fps: int = 16
    seed: int = 42
    sample_shift: float = 10.0
    max_area: int = 480 * 832
    max_sequence_length: int = 512
    local_attn_size: int = 18
    sink_size: int = 6
    warmup_chunks: int = 1
    dit_device: str = "cuda:0"
    vae_device: str = "cuda:0"
    text_device: str = "cuda:0"
    save_video: bool = False


@dataclass(frozen=True)
class GenerationLoopResult:
    chunks: list[dict[str, float | int]]
    total_frames: int
    encode_seconds: float


@dataclass(frozen=True)
class BenchmarkCase:
    image_path: Path
    action_path: Path
    prompt: str


def parse_config(argv: Sequence[str] | None = None) -> BenchmarkConfig:
    model_zoo = Path(os.environ.get("TF_MODEL_ZOO_PATH", "model_zoo")).expanduser()
    parser = argparse.ArgumentParser(description="Benchmark TeleFuser LingBot World Fast chunk generation.")
    parser.add_argument("--checkpoint-dir", type=Path, default=model_zoo / "Wan2.2-I2V-A14B")
    parser.add_argument("--fast-checkpoint-dir", type=Path, default=model_zoo / "lingbot-world-fast")
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--frame-num", type=int, default=201)
    parser.add_argument("--chunk-size", type=int, default=3)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-shift", type=float, default=10.0)
    parser.add_argument("--max-area", type=int, default=480 * 832)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--local-attn-size", type=int, default=18)
    parser.add_argument("--sink-size", type=int, default=6)
    parser.add_argument("--warmup-chunks", type=int, default=1)
    parser.add_argument("--dit-device", default="cuda:0")
    parser.add_argument("--vae-device", default="cuda:0")
    parser.add_argument("--text-device", default="cuda:0")
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args(argv)
    return BenchmarkConfig(**vars(args))


def validate_attention_config(local_attn_size: int, sink_size: int) -> None:
    if local_attn_size < -1:
        raise ValueError("--local-attn-size must be -1 or non-negative")
    if sink_size < 0:
        raise ValueError("--sink-size must be non-negative")
    if local_attn_size == -1 and sink_size:
        raise ValueError("full attention requires --sink-size 0")
    if local_attn_size >= 0 and sink_size >= local_attn_size:
        raise ValueError("--sink-size must be smaller than local-attn-size")


def validate_case_dir(case_dir: Path, *, frame_num: int) -> BenchmarkCase:
    case_dir = case_dir.expanduser().resolve()
    image_path = case_dir / "image.jpg"
    prompt_path = case_dir / "prompt.txt"
    poses_path = case_dir / "poses.npy"
    intrinsics_path = case_dir / "intrinsics.npy"
    for path in (image_path, prompt_path, poses_path, intrinsics_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing benchmark input: {path}")
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Benchmark prompt is empty: {prompt_path}")
    poses = np.load(poses_path, mmap_mode="r")
    intrinsics = np.load(intrinsics_path, mmap_mode="r")
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected poses.npy shape (frames, 4, 4), got {poses.shape}")
    if intrinsics.ndim != 2 or intrinsics.shape[1:] != (4,):
        raise ValueError(f"Expected intrinsics.npy shape (frames, 4), got {intrinsics.shape}")
    if poses.shape[0] < frame_num or intrinsics.shape[0] < frame_num:
        raise ValueError(f"Camera controls must contain at least {frame_num} frames")
    return BenchmarkCase(image_path=image_path, action_path=case_dir, prompt=prompt)


def parse_runtime_device(device: str) -> tuple[str, int]:
    if device == "cuda":
        return "cuda", 0
    if device.startswith("cuda:"):
        try:
            return "cuda", int(device.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device: {device}") from exc
    if device == "cpu":
        return "cpu", 0
    raise ValueError(f"Unsupported runtime device: {device}")


def _synchronize_devices(torch_module: Any, devices: tuple[str, ...]) -> None:
    if not torch_module.cuda.is_available():
        return
    for device in devices:
        torch_module.cuda.synchronize(device)


def measure_phase(
    name: str,
    operation: Callable[[], Any],
    *,
    torch_module: Any,
    devices: tuple[str, ...],
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[Any, dict[str, Any]]:
    del name
    _synchronize_devices(torch_module, devices)
    for device in devices:
        torch_module.cuda.reset_peak_memory_stats(device)
    start = clock()
    value = operation()
    _synchronize_devices(torch_module, devices)
    seconds = clock() - start
    memory = [
        {
            "device": device,
            "peak_allocated_bytes": int(torch_module.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch_module.cuda.max_memory_reserved(device)),
        }
        for device in devices
    ]
    return value, {"seconds": seconds, "memory": memory}


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_chunks(
    chunks: list[dict[str, float | int]],
    *,
    warmup_chunks: int,
) -> dict[str, float | int]:
    skipped = min(max(0, warmup_chunks), len(chunks))
    steady = chunks[skipped:]
    if not steady:
        return {"count": 0, "warmup_chunks_skipped": skipped}
    durations = [float(chunk["compute_seconds"]) for chunk in steady]
    total_frames = sum(int(chunk["frames"]) for chunk in steady)
    total_seconds = sum(durations)
    return {
        "count": len(steady),
        "warmup_chunks_skipped": skipped,
        "mean_seconds": statistics.fmean(durations),
        "p50_seconds": _percentile(durations, 0.5),
        "p90_seconds": _percentile(durations, 0.9),
        "min_seconds": min(durations),
        "max_seconds": max(durations),
        "std_seconds": statistics.pstdev(durations),
        "frames_per_second": total_frames / total_seconds,
    }


def run_generation_loop(
    pipeline,
    runtime,
    *,
    synchronize: Callable[[], None],
    clock: Callable[[], float] = time.perf_counter,
    frame_sink: Callable[[list[object]], None] | None = None,
) -> GenerationLoopResult:
    chunks: list[dict[str, float | int]] = []
    total_frames = 0
    encode_seconds = 0.0
    while runtime.active:
        index = len(chunks)
        synchronize()
        start = clock()
        frames = pipeline.generate_next_chunk(runtime)
        synchronize()
        compute_seconds = clock() - start
        chunks.append({"index": index, "frames": len(frames), "compute_seconds": compute_seconds})
        total_frames += len(frames)
        if frame_sink is not None:
            encode_start = clock()
            frame_sink(frames)
            encode_seconds += clock() - encode_start
    return GenerationLoopResult(chunks=chunks, total_frames=total_frames, encode_seconds=encode_seconds)


def _jsonable_config(config: BenchmarkConfig) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}


def build_result_document(
    config: BenchmarkConfig,
    *,
    runtime: Any,
    loop: GenerationLoopResult,
    phases: dict[str, dict[str, Any]],
    environment: dict[str, Any],
    output_dir: Path,
    video_path: Path | None,
) -> dict[str, Any]:
    cache_capacity = int(runtime.self_kv_cache[0]["k"].shape[1])
    return {
        "schema_version": 1,
        "benchmark": "lingbot_world_fast",
        "status": "completed",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": _jsonable_config(config),
        "environment": environment,
        "runtime": {
            "height": int(runtime.height),
            "width": int(runtime.width),
            "latent_frames": int(runtime.latent_f),
            "frame_tokens": int(runtime.frame_tokens),
            "chunk_size": int(runtime.chunk_size),
            "max_attention_size": int(runtime.max_attention_size),
            "kv_local_attn_size": int(runtime.kv_local_attn_size),
            "kv_sink_size": int(runtime.kv_sink_size),
            "kv_cache_capacity_tokens": cache_capacity,
        },
        "timings": {
            "phases": phases,
            "chunks": loop.chunks,
            "steady_state": summarize_chunks(loop.chunks, warmup_chunks=config.warmup_chunks),
            "video_encode_seconds": loop.encode_seconds,
        },
        "total_frames": loop.total_frames,
        "artifacts": {
            "output_dir": str(output_dir),
            "video_path": str(video_path) if video_path is not None else None,
        },
    }


def _cuda_devices(config: BenchmarkConfig) -> tuple[str, ...]:
    devices: list[str] = []
    for value in (config.dit_device, config.vae_device, config.text_device):
        device_type, device_id = parse_runtime_device(value)
        if device_type == "cuda":
            canonical = f"cuda:{device_id}"
            if canonical not in devices:
                devices.append(canonical)
    return tuple(devices)


def _resolve_output_dir(config: BenchmarkConfig) -> Path:
    if config.output_dir is not None:
        return config.output_dir.expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_ROOT / timestamp


def run_benchmark(config: BenchmarkConfig, *, backend: Any | None = None) -> Path:
    validate_attention_config(config.local_attn_size, config.sink_size)
    if config.frame_num <= 0:
        raise ValueError("--frame-num must be positive")
    if config.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if config.fps <= 0:
        raise ValueError("--fps must be positive")
    if config.warmup_chunks < 0:
        raise ValueError("--warmup-chunks must be non-negative")
    case = validate_case_dir(config.case_dir, frame_num=config.frame_num)
    backend = backend or DefaultBenchmarkBackend()
    if hasattr(backend, "validate_model_paths"):
        backend.validate_model_paths(config)

    output_dir = _resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    devices = _cuda_devices(config)
    phases: dict[str, dict[str, Any]] = {}

    pipeline, phases["pipeline_init"] = measure_phase(
        "pipeline_init",
        lambda: backend.build_pipeline(config),
        torch_module=backend.torch,
        devices=devices,
    )
    runtime, phases["runtime_creation"] = measure_phase(
        "runtime_creation",
        lambda: backend.create_runtime(pipeline, config, case),
        torch_module=backend.torch,
        devices=devices,
    )

    video_path = output_dir / "video.mp4" if config.save_video else None
    video_sink = backend.open_video_sink(video_path, config.fps) if video_path is not None else None
    try:
        loop, phases["generation"] = measure_phase(
            "generation",
            lambda: run_generation_loop(
                pipeline,
                runtime,
                synchronize=lambda: _synchronize_devices(backend.torch, devices),
                frame_sink=video_sink,
            ),
            torch_module=backend.torch,
            devices=devices,
        )
    finally:
        if video_sink is not None:
            video_sink.close()

    result = build_result_document(
        config,
        runtime=runtime,
        loop=loop,
        phases=phases,
        environment=backend.collect_environment(devices),
        output_dir=output_dir,
        video_path=video_path,
    )
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result_path


class _VideoSink:
    def __init__(self, output_path: Path, fps: int) -> None:
        import imageio.v2 as imageio

        self._writer = imageio.get_writer(
            str(output_path),
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            quality=8,
            macro_block_size=2,
            ffmpeg_params=["-movflags", "+faststart"],
        )

    def __call__(self, frames: list[object]) -> None:
        for frame in frames:
            self._writer.append_data(np.asarray(frame.convert("RGB")))

    def close(self) -> None:
        self._writer.close()


class DefaultBenchmarkBackend:
    def __init__(self) -> None:
        import torch

        self.torch = torch

    @staticmethod
    def validate_model_paths(config: BenchmarkConfig) -> None:
        required = (
            config.checkpoint_dir / "Wan2.1_VAE.pth",
            config.checkpoint_dir / "models_t5_umt5-xxl-enc-bf16.pth",
            config.checkpoint_dir / "google" / "umt5-xxl",
            config.fast_checkpoint_dir,
        )
        missing = [path for path in required if not path.exists()]
        if missing:
            formatted = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(f"Missing required model paths:\n{formatted}")

    def build_pipeline(self, config: BenchmarkConfig):
        from telefuser.core.config import ModelRuntimeConfig
        from telefuser.core.module_manager import ModuleManager
        from telefuser.pipelines.lingbot_world_fast.pipeline import (
            LingBotWorldFastPipeline,
            LingBotWorldFastPipelineConfig,
        )

        vae_type, vae_id = parse_runtime_device(config.vae_device)
        text_type, text_id = parse_runtime_device(config.text_device)
        pipeline = LingBotWorldFastPipeline(device=config.dit_device, torch_dtype=self.torch.bfloat16)
        pipeline.init(
            ModuleManager(torch_dtype=self.torch.bfloat16, device="cpu"),
            LingBotWorldFastPipelineConfig(
                checkpoint_dir=str(config.checkpoint_dir),
                fast_checkpoint_subdir=str(config.fast_checkpoint_dir),
                vae_config=ModelRuntimeConfig(
                    device_type=vae_type,
                    device_id=vae_id,
                    torch_dtype=self.torch.bfloat16,
                ),
                text_encoding_config=ModelRuntimeConfig(
                    device_type=text_type,
                    device_id=text_id,
                    torch_dtype=self.torch.bfloat16,
                ),
                dit_torch_dtype=self.torch.bfloat16,
                control_type="cam",
                max_area=config.max_area,
                local_attn_size=config.local_attn_size,
                sink_size=config.sink_size,
            ),
        )
        return pipeline

    @staticmethod
    def create_runtime(pipeline: Any, config: BenchmarkConfig, case: BenchmarkCase):
        from PIL import Image

        from telefuser.pipelines.lingbot_world_fast.session import LingBotWorldFastSessionConfig

        return pipeline.create_runtime(
            LingBotWorldFastSessionConfig(
                prompt=case.prompt,
                image=Image.open(case.image_path).convert("RGB"),
                control_mode="cam",
                fps=config.fps,
                chunk_size=config.chunk_size,
                frame_num=config.frame_num,
                sample_shift=config.sample_shift,
                seed=config.seed,
                max_attention_size=None,
                max_sequence_length=config.max_sequence_length,
                action_path=str(case.action_path),
                show_control_hud=False,
            )
        )

    @staticmethod
    def open_video_sink(output_path: Path, fps: int) -> _VideoSink:
        return _VideoSink(output_path, fps)

    def collect_environment(self, devices: tuple[str, ...]) -> dict[str, Any]:
        try:
            commit = subprocess.run(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        gpu_info = []
        if self.torch.cuda.is_available():
            for device in devices:
                properties = self.torch.cuda.get_device_properties(device)
                gpu_info.append(
                    {
                        "device": device,
                        "name": properties.name,
                        "compute_capability": f"{properties.major}.{properties.minor}",
                        "total_memory_bytes": int(properties.total_memory),
                    }
                )
        return {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": self.torch.__version__,
            "cuda_version": self.torch.version.cuda,
            "telefuser_git_commit": commit,
            "gpus": gpu_info,
        }


def main(argv: Sequence[str] | None = None) -> int:
    result_path = run_benchmark(parse_config(argv))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    steady = result["timings"]["steady_state"]
    print(f"Result: {result_path}")
    print(f"Frames: {result['total_frames']}")
    if steady["count"]:
        print(
            f"Steady-state: mean={steady['mean_seconds']:.4f}s "
            f"p90={steady['p90_seconds']:.4f}s fps={steady['frames_per_second']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
