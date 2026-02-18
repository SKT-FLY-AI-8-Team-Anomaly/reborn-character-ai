"""
FastAPI 백엔드 (AI 서버): 사진 → AI 프로필, 프로필 → 모션 시트 1장 업로드.

POST /profile: multipart(image, uploadUrl, blobUrl) → 프로필 1장 생성 후 uploadUrl로 PUT.
POST /motion: JSON 수신 즉시 200 반환, 실제 작업은 백그라운드 → 완료/실패 시 callbackUrl로 웹훅 POST.
"""
import asyncio
import logging

logger = logging.getLogger(__name__)

# 모션 작업 타임아웃 (초). 이 시간 초과 시 실패 처리 후 콜백.
MOTION_JOB_TIMEOUT = 600
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
import httpx

from config import GEMINI_API_KEY, GENERATOR_MODEL
from agents.generator_agent import generate_profile, generate_four_dir_8frame_sheet

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


# --- 스키마 ---

class MotionRequest(BaseModel):
    """POST /motion 요청: 즉시 200 반환, 백그라운드에서 시트 생성 후 uploadUrl PUT, 끝나면 callbackUrl로 웹훅."""
    profileUrl: str
    uploadUrl: str
    blobUrl: str
    callbackUrl: str
    jobId: str
    userId: int


# --- 라우트 ---

@app.get("/")
def root():
    return {"service": "Pixel Refiner API", "docs": "/docs"}


@app.post("/profile")
async def create_profile(
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
        image_bytes = await image.read()
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


def _put_png_to_url(url: str, png_bytes: bytes, timeout: float = 60.0) -> httpx.Response:
    """PNG 바이너리를 Azure Blob SAS URL로 PUT."""
    headers = {
        "Content-Type": "image/png",
        "Content-Length": str(len(png_bytes)),
        "x-ms-blob-type": "BlockBlob",
    }
    with httpx.Client(timeout=timeout) as client:
        return client.put(url, content=png_bytes, headers=headers)


def _generate_and_upload_motion_sheet(
    profile_bytes: bytes,
    image_mime: str,
    upload_url: str,
) -> None:
    """동기: 8프레임 시트 생성 후 upload_url로 PUT. 실패 시 예외."""
    sheet_bytes = generate_four_dir_8frame_sheet(
        profile_bytes,
        api_key=GEMINI_API_KEY,
        model=GENERATOR_MODEL,
        image_mime=image_mime,
    )
    r = _put_png_to_url(upload_url, sheet_bytes)
    if r.status_code >= 400:
        raise RuntimeError(f"Storage returned {r.status_code}: {r.text or r.reason_phrase}")


async def _post_callback(callback_url: str, job_id: str, user_id: int, success: bool) -> None:
    """웹훅 콜백 1회 POST."""
    payload = {"jobId": job_id, "userId": user_id, "success": success}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(callback_url, json=payload)
        logger.info("motion callback sent jobId=%s success=%s status=%s", job_id, success, r.status_code)
    except Exception as e:
        logger.warning("motion callback failed jobId=%s url=%s err=%s", job_id, callback_url, e)


async def _run_motion_job(body: MotionRequest) -> None:
    """백그라운드: 프로필 다운로드 → 시트 생성·PUT → callbackUrl POST."""
    job_id = body.jobId
    try:
        logger.info("motion job started jobId=%s", job_id)
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(body.profileUrl)
        if resp.status_code >= 400 or not resp.content:
            logger.warning("motion job jobId=%s profile download failed status=%s", job_id, resp.status_code)
            await _post_callback(body.callbackUrl, body.jobId, body.userId, success=False)
            return
        profile_bytes = resp.content
        image_mime = resp.headers.get("content-type") or "image/png"
        logger.info("motion job jobId=%s profile downloaded, generating sheet...", job_id)

        loop = asyncio.get_event_loop()
        await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: _generate_and_upload_motion_sheet(
                    profile_bytes, image_mime, body.uploadUrl
                ),
            ),
            timeout=MOTION_JOB_TIMEOUT,
        )
        logger.info("motion job jobId=%s sheet generated and uploaded", job_id)
        await _post_callback(body.callbackUrl, body.jobId, body.userId, success=True)
    except asyncio.TimeoutError:
        logger.error("motion job jobId=%s timed out after %ss", job_id, MOTION_JOB_TIMEOUT)
        await _post_callback(body.callbackUrl, body.jobId, body.userId, success=False)
    except Exception as e:
        logger.exception("motion job jobId=%s failed: %s", job_id, e)
        await _post_callback(body.callbackUrl, body.jobId, body.userId, success=False)


@app.post("/motion")
async def create_motion(body: MotionRequest):
    """
    요청 수신 즉시 200 OK 반환. 실제 작업은 백그라운드에서 진행하며,
    완료/실패 시 callbackUrl로 POST: { "jobId", "userId", "success": true|false }.
    """
    if not body.profileUrl.strip():
        raise HTTPException(400, "profileUrl is required")
    if not body.uploadUrl.strip():
        raise HTTPException(400, "uploadUrl is required")
    if not body.callbackUrl.strip():
        raise HTTPException(400, "callbackUrl is required")

    asyncio.create_task(_run_motion_job(body))
    return Response(status_code=200)
