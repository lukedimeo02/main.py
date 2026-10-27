"""
Face Analysis API
=================
Simple FastAPI service that accepts face photos and returns
structured measurements (facial thirds, symmetry, ratios, scores).

Endpoints:
  POST /analyze          - Upload one or more photos, get analysis
  GET  /health           - Health check
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from face_processor import FaceProcessor, MeasurementResult

app = FastAPI(
    title="Face Analysis API",
    description="Qoves-style facial measurements using MediaPipe",
    version="1.0.0",
)

# Allow calls from Base44 and local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create the processor once at startup (loads the model)
processor: Optional[FaceProcessor] = None


@app.on_event("startup")
def load_model():
    global processor
    processor = FaceProcessor()
    print("FaceProcessor loaded successfully")


@app.on_event("shutdown")
def shutdown():
    if processor:
        processor.close()


class AnalysisResponse(BaseModel):
    success: bool
    message: str
    scores: dict = {}
    facial_thirds: dict = {}
    eye_spacing_ratio: float = 0.0
    symmetry_score: float = 0.0
    nose_width_ratio: float = 0.0
    lip_ratio: float = 0.0
    face_width_to_height: float = 0.0
    midface_ratio: float = 0.0
    jaw_angle_left: Optional[float] = None
    jaw_angle_right: Optional[float] = None
    image_width: int = 0
    image_height: int = 0
    # Full raw data if you need it
    full_result: dict = {}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": processor is not None}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_face(
    front: UploadFile = File(..., description="Front-facing face photo (required)"),
    left_profile: Optional[UploadFile] = File(None, description="Optional left profile"),
    right_profile: Optional[UploadFile] = File(None, description="Optional right profile"),
):
    """
    Analyze a face from uploaded photos.
    Only the front photo is required for the current measurements.
    """
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Validate content type roughly
    if front.content_type and not front.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Front file must be an image")

    try:
        # Save front photo to a temporary file
        front_bytes = await front.read()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(front_bytes)
            front_path = tmp.name

        # Optional profiles (currently not used in measurements, but accepted)
        left_path = None
        right_path = None

        if left_profile:
            left_bytes = await left_profile.read()
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(left_bytes)
                left_path = tmp.name

        if right_profile:
            right_bytes = await right_profile.read()
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(right_bytes)
                right_path = tmp.name

        # Run analysis
        result: MeasurementResult = processor.analyze(
            front_path=front_path,
            profile_left=left_path,
            profile_right=right_path,
        )

        # Clean up temp files
        for p in [front_path, left_path, right_path]:
            if p and Path(p).exists():
                Path(p).unlink(missing_ok=True)

        return AnalysisResponse(
            success=result.success,
            message=result.message,
            scores=result.scores,
            facial_thirds=result.facial_thirds,
            eye_spacing_ratio=result.eye_spacing_ratio,
            symmetry_score=result.symmetry_score,
            nose_width_ratio=result.nose_width_ratio,
            lip_ratio=result.lip_ratio,
            face_width_to_height=result.face_width_to_height,
            midface_ratio=result.midface_ratio,
            jaw_angle_left=result.jaw_angle_left,
            jaw_angle_right=result.jaw_angle_right,
            image_width=result.image_width,
            image_height=result.image_height,
            full_result=result.to_dict(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# Simple root
@app.get("/")
def root():
    return {
        "service": "Face Analysis API",
        "docs": "/docs",
        "health": "/health",
        "analyze": "POST /analyze",
    }
