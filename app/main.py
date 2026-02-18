"""
FastAPI 백엔드 (AI 서버): 사진 → AI 프로필, AI 프로필 → 4모션 스프라이트.

POST /profile: multipart(image, uploadUrl, blobUrl) → 프로필 1장 생성 후 uploadUrl로 PUT.
"""
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import httpx

from config import GEMINI_API_KEY, GENERATOR_MODEL
from agents.generator_agent import generate_profile

app = FastAPI(
    title="Pixel Refiner API",
    description="사진 → AI 프로필 생성, AI 프로필 → 4방향 스프라이트 시트 생성",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 스키마 (sprites는 추후 구현) ---

class SpritesRequest(BaseModel):
    """4모션 생성 요청 (AI 프로필 식별자 또는 이미지 등)."""
    profile_id: Optional[str] = None


class SpritesResponse(BaseModel):
    """4모션 스프라이트 생성 결과 (스켈레톤)."""
    success: bool = True
    sprite_sheet_url: Optional[str] = None
    message: str = "Sprites endpoint: not implemented yet"


# --- 라우트 ---

@app.get("/")
def root():
    return {"service": "Pixel Refiner API", "docs": "/docs"}


@app.post("/profile")
def create_profile(
    image: UploadFile = File(..., description="사용자 원본 이미지 (image.png 등)"),
    uploadUrl: str = Form(..., description="SAS 포함, 생성된 프로필 이미지를 PUT할 URL"),
    blobUrl: str = Form(..., description="SAS 없음, 업로드 후 읽을 때 쓸 URL (참고용)"),
):
    """
    multipart/form-data: image, uploadUrl, blobUrl.
    원본 이미지로 프로필 1장 생성 후 uploadUrl로 PUT. 성공 시 200 + {"ok": true}.
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "image file required")

    try:
        image_bytes = image.read()
    except Exception as e:
        raise HTTPException(400, f"Failed to read image: {e}") from e
    if not image_bytes:
        raise HTTPException(400, "image file is empty")

    if not uploadUrl.strip():
        raise HTTPException(400, "uploadUrl is required")

    # 프로필 1장 생성 (동기, 블로킹)
    try:
        profile_png = generate_profile(
            image_bytes,
            api_key=GEMINI_API_KEY,
            model=GENERATOR_MODEL,
            image_mime=image.content_type or "image/png",
        )
    except Exception as e:
        raise HTTPException(502, f"Profile generation failed: {e}") from e

    # Azure Blob에 PUT
    headers = {
        "Content-Type": "image/png",
        "Content-Length": str(len(profile_png)),
        "x-ms-blob-type": "BlockBlob",
    }
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.put(uploadUrl, content=profile_png, headers=headers)
    except Exception as e:
        raise HTTPException(502, f"Upload to storage failed: {e}") from e

    if resp.status_code >= 400:
        raise HTTPException(
            resp.status_code,
            f"Storage returned {resp.status_code}: {resp.text or resp.reason_phrase}",
        )

    return JSONResponse(content={"ok": True}, status_code=200)


@app.post("/sprites", response_model=SpritesResponse)
async def create_sprites(body: SpritesRequest):
    """
    AI 프로필을 받아 4방향 모션 스프라이트 시트를 생성합니다.
    (실제 생성 로직은 추후 연동)
    """
    return SpritesResponse(
        sprite_sheet_url=None,
        message="Sprites creation will be implemented; profile_id=" + str(body.profile_id),
    )
