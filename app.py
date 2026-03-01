from __future__ import annotations

import os
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from typing import Dict, List, Optional

from speaker import SpeakerBiometry

DB_PATH = os.path.join(os.path.dirname(__file__), "db", "profiles.npz")

app = FastAPI(
    title="Speaker Identification API",
    description="API for speaker enrollment and identification using voice biometry",
    version="1.0.0",
)

biometry = SpeakerBiometry(db_path=DB_PATH)


class EnrollResponse(BaseModel):
    status: str
    speaker_id: str
    total_speakers: int


class SpeakerScore(BaseModel):
    speaker_id: str
    score: float


class IdentifyResponse(BaseModel):
    scores: List[SpeakerScore]


class SpeakersResponse(BaseModel):
    speakers: List[str]


@app.post("/process", response_model=None)
async def process(
    file: UploadFile = File(..., description="Audio file (wav, ogg, mp3)"),
    speaker_id: Optional[str] = Form(None, description="Speaker ID for enrollment. Omit for identification."),
):
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        if speaker_id:
            result = biometry.enroll(speaker_id, tmp_path)
            return EnrollResponse(**result)
        else:
            scores = biometry.identify(tmp_path)
            return IdentifyResponse(
                scores=[SpeakerScore(speaker_id=sid, score=s) for sid, s in scores]
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.get("/speakers", response_model=SpeakersResponse)
async def list_speakers():
    return SpeakersResponse(speakers=list(biometry.db.keys()))


@app.get("/")
async def root():
    return {
        "message": "Speaker Identification API",
        "endpoints": {
            "POST /process": "Upload audio file. With speaker_id — enroll, without — identify.",
            "GET /speakers": "List enrolled speakers.",
        },
    }
