from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

BENCHMARK_PATH = Path(__file__).resolve().parents[3] / "benchmarks" / "lingbot_world_fast" / "benchmark.py"


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("lingbot_world_fast_benchmark", BENCHMARK_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_case(case_dir: Path, *, frames: int = 201) -> None:
    case_dir.mkdir()
    (case_dir / "image.jpg").write_bytes(b"image")
    (case_dir / "prompt.txt").write_text("A benchmark prompt.\n", encoding="utf-8")
    np.save(case_dir / "poses.npy", np.zeros((frames, 4, 4), dtype=np.float32))
    np.save(case_dir / "intrinsics.npy", np.zeros((frames, 4), dtype=np.float32))


def test_cli_defaults_and_device_parsing(monkeypatch) -> None:
    benchmark = load_benchmark_module()
    monkeypatch.setenv("TF_MODEL_ZOO_PATH", "/models")

    config = benchmark.parse_config([])

    assert config.checkpoint_dir == Path("/models/Wan2.2-I2V-A14B")
    assert config.fast_checkpoint_dir == Path("/models/lingbot-world-fast")
    assert (config.frame_num, config.chunk_size) == (201, 3)
    assert (config.local_attn_size, config.sink_size) == (18, 6)
    assert config.warmup_chunks == 1
    assert (config.dit_device, config.vae_device, config.text_device) == ("cuda:0", "cuda:0", "cuda:0")
    assert config.save_video is False
    assert benchmark.parse_runtime_device("cuda") == ("cuda", 0)
    assert benchmark.parse_runtime_device("cuda:3") == ("cuda", 3)
    assert benchmark.parse_runtime_device("cpu") == ("cpu", 0)


def test_attention_validation_rejects_invalid_combinations() -> None:
    benchmark = load_benchmark_module()
    invalid = [
        (-2, 0, "local-attn-size"),
        (-1, 1, "full attention"),
        (18, -1, "sink-size"),
        (18, 18, "smaller than local-attn-size"),
    ]

    for local_attn_size, sink_size, match in invalid:
        with pytest.raises(ValueError, match=match):
            benchmark.validate_attention_config(local_attn_size, sink_size)


def test_chunk_summary_excludes_warmup() -> None:
    benchmark = load_benchmark_module()
    chunks = [
        {"index": 0, "frames": 9, "compute_seconds": 10.0},
        {"index": 1, "frames": 12, "compute_seconds": 2.0},
        {"index": 2, "frames": 12, "compute_seconds": 4.0},
        {"index": 3, "frames": 12, "compute_seconds": 6.0},
    ]

    summary = benchmark.summarize_chunks(chunks, warmup_chunks=1)

    assert summary == {
        "count": 3,
        "warmup_chunks_skipped": 1,
        "mean_seconds": pytest.approx(4.0),
        "p50_seconds": pytest.approx(4.0),
        "p90_seconds": pytest.approx(5.6),
        "min_seconds": pytest.approx(2.0),
        "max_seconds": pytest.approx(6.0),
        "std_seconds": pytest.approx(1.632993161855452),
        "frames_per_second": pytest.approx(3.0),
    }


def test_generation_loop_keeps_video_encoding_out_of_compute_time() -> None:
    benchmark = load_benchmark_module()
    written: list[int] = []
    sync_calls: list[None] = []

    result = benchmark.run_generation_loop(
        FakePipeline(),
        FakeRuntime(),
        synchronize=lambda: sync_calls.append(None),
        clock=FakeClock([1.0, 3.0, 3.0, 4.0, 10.0, 15.0, 15.0, 17.0]),
        frame_sink=lambda frames: written.append(len(frames)),
    )

    assert result.total_frames == 21
    assert result.encode_seconds == pytest.approx(3.0)
    assert result.chunks == [
        {"index": 0, "frames": 9, "compute_seconds": 2.0},
        {"index": 1, "frames": 12, "compute_seconds": 5.0},
    ]
    assert written == [9, 12]
    assert len(sync_calls) == 4


def test_case_validation_checks_shapes_and_length(tmp_path) -> None:
    benchmark = load_benchmark_module()
    case_dir = tmp_path / "case03"
    write_case(case_dir)

    case = benchmark.validate_case_dir(case_dir, frame_num=201)

    assert case.image_path == case_dir / "image.jpg"
    assert case.action_path == case_dir
    assert case.prompt == "A benchmark prompt."

    np.save(case_dir / "poses.npy", np.zeros((200, 4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="at least 201 frames"):
        benchmark.validate_case_dir(case_dir, frame_num=201)


def test_measure_phase_initializes_cuda_before_resetting_memory_stats() -> None:
    benchmark = load_benchmark_module()
    fake_torch = FakeTorch()

    value, measurement = benchmark.measure_phase(
        "pipeline_init",
        lambda: "pipeline",
        torch_module=fake_torch,
        devices=("cuda:0", "cuda:1"),
        clock=FakeClock([2.0, 5.5]),
    )

    assert value == "pipeline"
    assert measurement["seconds"] == pytest.approx(3.5)
    assert measurement["memory"] == [
        {"device": "cuda:0", "peak_allocated_bytes": 100, "peak_reserved_bytes": 200},
        {"device": "cuda:1", "peak_allocated_bytes": 101, "peak_reserved_bytes": 201},
    ]
    assert fake_torch.cuda.reset_devices == ["cuda:0", "cuda:1"]
    assert fake_torch.cuda.sync_devices == ["cuda:0", "cuda:1", "cuda:0", "cuda:1"]


def test_fake_benchmark_writes_complete_report_without_video(tmp_path) -> None:
    benchmark = load_benchmark_module()
    case_dir = tmp_path / "case03"
    write_case(case_dir)
    output_dir = tmp_path / "output"
    backend = FakeBackend()
    config = benchmark.BenchmarkConfig(
        checkpoint_dir=tmp_path / "base",
        fast_checkpoint_dir=tmp_path / "fast",
        case_dir=case_dir,
        output_dir=output_dir,
    )

    result_path = benchmark.run_benchmark(config, backend=backend)

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result_path == output_dir / "result.json"
    assert (result["schema_version"], result["status"], result["total_frames"]) == (1, "completed", 21)
    assert set(result["timings"]["phases"]) == {"pipeline_init", "runtime_creation", "generation"}
    assert result["timings"]["steady_state"]["count"] == 1
    assert result["runtime"]["kv_local_attn_size"] == 18
    assert result["runtime"]["kv_sink_size"] == 6
    assert result["runtime"]["kv_cache_capacity_tokens"] == 2340
    assert result["environment"] == {"backend": "fake"}
    assert result["artifacts"]["video_path"] is None
    assert backend.video_opened is False


class FakeRuntime:
    def __init__(self) -> None:
        self.active = True
        self.current_chunk_index = 0


class FakePipeline:
    def generate_next_chunk(self, runtime: FakeRuntime):
        frame_counts = (9, 12)
        count = frame_counts[runtime.current_chunk_index]
        runtime.current_chunk_index += 1
        runtime.active = runtime.current_chunk_index < len(frame_counts)
        return [object()] * count


class FakeClock:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class FakeCuda:
    def __init__(self) -> None:
        self.reset_devices: list[str] = []
        self.sync_devices: list[str] = []
        self.initialized_devices: set[str] = set()

    @staticmethod
    def is_available() -> bool:
        return True

    def reset_peak_memory_stats(self, device: str) -> None:
        if device not in self.initialized_devices:
            raise RuntimeError("CUDA context is not initialized")
        self.reset_devices.append(device)

    def synchronize(self, device: str) -> None:
        self.initialized_devices.add(device)
        self.sync_devices.append(device)

    @staticmethod
    def max_memory_allocated(device: str) -> int:
        return 100 + int(device.rsplit(":", 1)[1])

    @staticmethod
    def max_memory_reserved(device: str) -> int:
        return 200 + int(device.rsplit(":", 1)[1])


class FakeTorch:
    def __init__(self) -> None:
        self.cuda = FakeCuda()


class FakeBenchmarkRuntime(FakeRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.frame_tokens = 130
        self.chunk_size = 3
        self.max_attention_size = 2340
        self.latent_f = 51
        self.height = 480
        self.width = 832
        self.kv_local_attn_size = 18
        self.kv_sink_size = 6
        self.self_kv_cache = [{"k": SimpleNamespace(shape=(1, 2340, 40, 128))}]


class FakeBackend:
    def __init__(self) -> None:
        self.torch = FakeTorch()
        self.video_opened = False

    @staticmethod
    def build_pipeline(config):
        return FakePipeline()

    @staticmethod
    def create_runtime(pipeline, config, case):
        return FakeBenchmarkRuntime()

    @staticmethod
    def collect_environment(devices):
        return {"backend": "fake"}

    def open_video_sink(self, output_path: Path, fps: int):
        self.video_opened = True
        raise AssertionError("Video sink must not be opened by default")
