from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import torch

from telefuser.core.config import ModelRuntimeConfig
from telefuser.models.wan_video_vae import VideoVAE
from telefuser.pipelines.lingbot_world_fast.denoising import LingBotWorldFastDenoisingStage
from telefuser.pipelines.lingbot_world_fast.vae_stage import LingBotWorldFastVAEEncodeStage


@dataclass
class _RecordingVAE:
    geometry_calls: int = 0
    frame_counts: list[int] = field(default_factory=list)

    def _encoder_temporal_geometry(self) -> tuple[int, int]:
        self.geometry_calls += 1
        return 113, 4

    def cached_encode_withflag(
        self,
        video: torch.Tensor,
        device: torch.device,
        is_first_clip: bool,
        is_last_clip: bool,
        encode_state: object,
    ) -> torch.Tensor:
        del device, encode_state
        assert is_first_clip is True
        assert is_last_clip is True
        self.frame_counts.append(video.shape[1])
        latent_frames = (video.shape[1] - 1) // 4 + 1
        values = torch.arange(latent_frames, dtype=video.dtype).view(1, latent_frames, 1, 1)
        return values.expand(16, -1, 2, 2).clone()


@dataclass
class _FakeModuleManager:
    vae: _RecordingVAE

    def fetch_module(self, name: str) -> _RecordingVAE:
        assert name == "wan_video_vae"
        return self.vae


def _stage() -> LingBotWorldFastVAEEncodeStage:
    return LingBotWorldFastVAEEncodeStage(
        "test_vae_encode",
        _FakeModuleManager(_RecordingVAE()),
        ModelRuntimeConfig(device_type="cpu", torch_dtype=torch.float32),
    )


def _encode(
    stage: LingBotWorldFastVAEEncodeStage,
    cache_handle: int,
    chunk_index: int,
    chunk_count: int,
    chunk_size: int,
) -> dict[str, object]:
    encode = LingBotWorldFastVAEEncodeStage.encode_condition_chunk.__wrapped__
    return encode(stage, cache_handle, chunk_index, chunk_count, chunk_size, 2, 2, torch.float32)


def test_wan_encoder_temporal_geometry_is_derived_from_topology() -> None:
    default_model = VideoVAE(dim=2, z_dim=2)
    spatial_only_model = VideoVAE(dim=2, z_dim=2, temperal_downsample=[False, False, False])

    assert default_model._encoder_temporal_geometry() == (113, 4)
    assert spatial_only_model._encoder_temporal_geometry() == (45, 1)


def test_derived_prefix_reconstructs_the_complete_zero_tail() -> None:
    torch.manual_seed(0)
    model = VideoVAE(dim=2, z_dim=2).eval()
    video = torch.zeros(1, 3, 121, 8, 8)
    video[:, :, 0] = torch.randn(1, 3, 8, 8)
    scale = torch.stack([torch.zeros(2), torch.ones(2)])

    with torch.no_grad():
        complete = model.encode(video, scale)
        prefix = model.encode(video[:, :, :117], scale)
    reconstructed = torch.cat([prefix, prefix[:, :, -1:].expand(-1, -1, 1, -1, -1)], dim=2)

    assert complete.shape[2] == 31
    assert prefix.shape[2] == 30
    assert not torch.equal(complete[:, :, 15], complete[:, :, 29])
    assert torch.equal(reconstructed, complete)


def test_long_session_uses_the_first_input_independent_latent_as_its_tail() -> None:
    stage = _stage()
    assert stage._condition_prefix_latent_frames == 30
    assert stage.vae.geometry_calls == 1
    assert stage.initialize_cache(1, torch.ones(3, 2, 2)) is True

    first_packet = _encode(stage, 1, 0, 10, 4)
    tail_packet = _encode(stage, 1, 9, 10, 4)
    resolver = LingBotWorldFastDenoisingStage.__new__(LingBotWorldFastDenoisingStage)
    denoise_state = SimpleNamespace(image_condition_latent=None)
    resolver._resolve_image_condition(
        denoise_state,
        first_packet,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    tail = resolver._resolve_image_condition(
        denoise_state,
        tail_packet,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert stage.vae.frame_counts == [117]
    assert first_packet["latent_condition"].shape == (16, 30, 2, 2)
    assert tail_packet["latent_condition"] is None
    assert torch.equal(tail[0, 4:], torch.full((16, 4, 2, 2), 29.0))


def test_short_session_encodes_its_complete_condition_sequence() -> None:
    stage = _stage()
    assert stage.initialize_cache(1, torch.ones(3, 2, 2)) is True

    first_packet = _encode(stage, 1, 0, 5, 4)
    final_packet = _encode(stage, 1, 4, 5, 4)
    resolver = LingBotWorldFastDenoisingStage.__new__(LingBotWorldFastDenoisingStage)
    denoise_state = SimpleNamespace(image_condition_latent=None)
    resolver._resolve_image_condition(
        denoise_state,
        first_packet,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    final = resolver._resolve_image_condition(
        denoise_state,
        final_packet,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert stage.vae.frame_counts == [77]
    assert first_packet["latent_condition"].shape == (16, 20, 2, 2)
    assert torch.equal(final[0, 4:, :, 0, 0], torch.arange(16, 20).expand(16, -1))
