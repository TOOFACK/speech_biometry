from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition


@dataclass
class PreprocessConfig:
    target_sr: int = 16000
    vad_frame_ms: int = 30
    vad_hop_ms: int = 10
    vad_energy_threshold_db: float = -35.0
    vad_min_speech_ms: int = 300
    max_audio_s: float = 12.0
    min_audio_s: float = 0.8


def _to_mono(wav: torch.Tensor) -> torch.Tensor:
    if wav.dim() == 1:
        return wav
    if wav.size(0) == 1:
        return wav[0]
    return torch.mean(wav, dim=0)


def _frame_rms_db(x: np.ndarray, eps: float = 1e-12) -> float:
    rms = math.sqrt(float(np.mean(x * x)) + eps)
    return 20.0 * math.log10(rms + eps)


def _energy_vad(wav: torch.Tensor, sr: int, cfg: PreprocessConfig) -> torch.Tensor:
    x = wav.detach().cpu().numpy().astype(np.float32)
    frame = int(sr * cfg.vad_frame_ms / 1000.0)
    hop = int(sr * cfg.vad_hop_ms / 1000.0)
    if frame <= 0 or hop <= 0:
        return wav

    flags = []
    idxs = []
    for start in range(0, max(1, len(x) - frame + 1), hop):
        chunk = x[start : start + frame]
        db = _frame_rms_db(chunk)
        flags.append(db >= cfg.vad_energy_threshold_db)
        idxs.append((start, start + frame))

    if not flags:
        return wav

    segments = []
    cur_start = None
    cur_end = None
    for speech, (s, e) in zip(flags, idxs):
        if speech:
            if cur_start is None:
                cur_start = s
            cur_end = e
        else:
            if cur_start is not None:
                segments.append((cur_start, cur_end))
                cur_start = None
                cur_end = None
    if cur_start is not None:
        segments.append((cur_start, cur_end))

    if not segments:
        return torch.zeros(0, dtype=wav.dtype)

    min_len = int(sr * cfg.vad_min_speech_ms / 1000.0)
    kept = [x[s:e] for s, e in segments if (e - s) >= min_len]

    if not kept:
        return torch.zeros(0, dtype=wav.dtype)

    return torch.from_numpy(np.concatenate(kept, axis=0)).to(wav.dtype)


def preprocess(path: str, cfg: PreprocessConfig = PreprocessConfig()) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    wav = _to_mono(wav)
    wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=cfg.target_sr) if sr != cfg.target_sr else wav
    max_len = int(cfg.max_audio_s * cfg.target_sr)
    if wav.numel() > max_len:
        wav = wav[:max_len]
    wav = _energy_vad(wav, cfg.target_sr, cfg)
    if wav.numel() < int(cfg.min_audio_s * cfg.target_sr):
        return torch.zeros(0, dtype=torch.float32)
    wav = wav.float()
    peak = torch.max(torch.abs(wav)) + 1e-9
    return wav / peak


def l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x) + eps)


class SpeakerBiometry:
    VAD_THRESHOLDS = [-35.0, -40.0, -45.0, -50.0, -55.0]

    def __init__(self, db_path: str = "db/profiles.npz"):
        self.db_path = db_path
        self.model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
        )
        self.db = self._load_db()

    def _load_db(self) -> Dict[str, np.ndarray]:
        if not os.path.exists(self.db_path):
            return {}
        try:
            if os.path.getsize(self.db_path) == 0:
                return {}
            data = np.load(self.db_path, allow_pickle=True)
        except (EOFError, ValueError):
            return {}
        return {k: data[k].astype(np.float32) for k in data.files}

    def _save_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        np.savez(self.db_path, **self.db)

    @torch.inference_mode()
    def _extract_embedding(self, wav: torch.Tensor) -> np.ndarray:
        wav_b = wav.unsqueeze(0)
        emb = self.model.encode_batch(wav_b).squeeze().detach().cpu().numpy().astype(np.float32)
        return l2norm(emb)

    def _preprocess_auto(self, path: str) -> torch.Tensor:
        for th in self.VAD_THRESHOLDS:
            cfg = PreprocessConfig(vad_energy_threshold_db=th)
            wav = preprocess(path, cfg)
            if wav.numel() > 0:
                return wav
        return torch.zeros(0, dtype=torch.float32)

    def enroll(self, speaker_id: str, audio_path: str) -> Dict:
        wav = self._preprocess_auto(audio_path)
        if wav.numel() == 0:
            raise ValueError("Too little speech detected in audio. Try a longer or louder recording.")

        emb = self._extract_embedding(wav)

        if speaker_id in self.db:
            old = self.db[speaker_id]
            self.db[speaker_id] = l2norm((old + emb) / 2.0)
        else:
            self.db[speaker_id] = emb

        self._save_db()
        return {
            "status": "enrolled",
            "speaker_id": speaker_id,
            "total_speakers": len(self.db),
        }

    def identify(self, audio_path: str) -> List[Tuple[str, float]]:
        if not self.db:
            raise ValueError("Speaker database is empty. Enroll at least one speaker first.")

        wav = self._preprocess_auto(audio_path)
        if wav.numel() == 0:
            raise ValueError("Too little speech detected in audio. Try a longer or louder recording.")

        emb = self._extract_embedding(wav)

        scores = []
        for spk_id, profile in self.db.items():
            sim = float(np.dot(l2norm(emb), l2norm(profile)))
            scores.append((spk_id, round(sim, 4)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
