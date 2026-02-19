"""
FastAPI 백엔드 (AI 서버): 사진 → AI 프로필, 프로필 → 모션 시트 1장 업로드.

POST /profile: multipart(image, uploadUrl, blobUrl) → 프로필 1장 생성 후 uploadUrl로 PUT.
POST /motion: JSON 수신 즉시 200 반환, 백그라운드에서 6프레임 시트 생성·누끼 후 uploadUrl로 PUT, 완료/실패 시 callbackUrl 웹훅.
"""
import asyncio
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# uvicorn과 같은 터미널에 [motion] 등 앱 로그가 찍히도록 설정 (한 번만)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setLevel(logging.INFO)
    _h.setFormatter(logging.Formatter("%(levelname)s:     [%(name)s] %(message)s"))
    logger.setLevel(logging.INFO)
    logger.addHandler(_h)

# 모션 작업 타임아웃 (초). 이 시간 초과 시 실패 처리 후 콜백.
MOTION_JOB_TIMEOUT = 600
# Gemini 503 시 재시도 횟수 및 대기(초). [5, 10, 20] → 최대 3회 재시도
MOTION_503_MAX_RETRIES = 3
MOTION_503_BACKOFF_SECS = [5, 10, 20]
# 처음 만든 6프레임 시트를 저장할 로컬 폴더
OUTPUTS_DIR = Path("outputs")
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
import httpx

from config import GEMINI_API_KEY, GENERATOR_MODEL
from agents.generator_agent import generate_profile, generate_six_frame_sheet_with_rembg

app = FastAPI(
    title="Pixel Refiner API",
    description="사진 → AI 프로필 생성, AI 프로필 → 6프레임 스프라이트 시트(누끼) 생성",
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
#
# [POST /motion 실행 흐름] 요청 들어온 순간부터 완료까지 호출되는 함수 순서:
#
#  1. create_motion(body: MotionRequest)  ← 진입점. FastAPI가 JSON body를 MotionRequest로 파싱 후 호출.
#     ├─ 로그: POST /motion received (jobId, userId, profileUrl, uploadUrl, callbackUrl)
#     ├─ validation: profileUrl, uploadUrl, callbackUrl 비어 있으면 400 예외
#     ├─ asyncio.create_task(_run_motion_job(body))  ← 백그라운드 태스크 생성 (비동기, 기다리지 않음)
#     └─ return Response(200)  ← 클라이언트에는 즉시 200 반환
#
#  2. _run_motion_job(body)  ← 별도 태스크에서 실행 (클라이언트는 이미 200 받은 뒤)
#     ├─ 로그: background job started
#     ├─ httpx.AsyncClient().get(profileUrl)  ← 프로필 이미지 다운로드
#     ├─ 실패 시(4xx/빈 body): _post_callback(..., success=False, error=...) → return
#     ├─ 성공 시: profile_bytes, image_mime 확보
#     ├─ loop.run_in_executor(None, lambda: _generate_and_upload_motion_sheet(...))  ← 스레드 풀에서 동기 실행, MOTION_JOB_TIMEOUT 초 대기
#     ├─ 성공 시: _post_callback(..., success=True)
#     └─ 예외 시(TimeoutError/Exception): _post_callback(..., success=False, error=err_msg)
#
#  3. _generate_and_upload_motion_sheet(profile_bytes, image_mime, upload_url, job_id)  ← run_in_executor 안에서 동기 호출
#     ├─ generate_six_frame_sheet_with_rembg(...)  ← agents/generator_agent: 6프레임 시트 생성 + remove.bg API 누끼
#     ├─ OUTPUTS_DIR에 {jobId}_6frame.png 저장
#     ├─ _put_png_to_url(upload_url, sheet_6frame)  ← SAS로 6프레임 PNG PUT
#     └─ 4xx 시 RuntimeError → _run_motion_job의 except에서 처리 후 콜백
#
#  4. _post_callback(callback_url, job_id, user_id, success, error?)  ← 성공/실패 시마다 1회 호출
#     ├─ payload = { jobId, userId, success [, error ] }
#     └─ httpx.AsyncClient().post(callback_url, json=payload)
#
#  5. _put_png_to_url(url, png_bytes)  ← _generate_and_upload_motion_sheet 내부에서만 사용
#     └─ httpx.Client().put(url, content=png_bytes, headers=...)
#

class MotionRequest(BaseModel):
    """POST /motion 요청: 즉시 200 반환, 백그라운드에서 6프레임 시트(누끼) 생성 후 uploadUrl PUT, 끝나면 callbackUrl 웹훅."""
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
    logger.info("POST /profile request received, content_type=%s", getattr(image, "content_type", None))
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
    """[흐름 5] PNG 바이너리를 Azure Blob SAS URL로 PUT. _generate_and_upload_motion_sheet 내부에서만 호출."""
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
    job_id: str,
) -> None:
    """[흐름 3] 동기: 6프레임 시트 생성 → 누끼 처리 → outputs 저장 후 upload_url로 6프레임 PNG PUT. run_in_executor에서 호출."""
    logger.info("[motion] jobId=%s _generate_and_upload_motion_sheet started (profile size=%s)", job_id, len(profile_bytes))
    logger.info("[motion] jobId=%s calling Gemini (6-frame sheet) + remove.bg... may take 1–2 min", job_id)
    sheet_6frame = generate_six_frame_sheet_with_rembg(
        profile_bytes,
        api_key=GEMINI_API_KEY,
        model=GENERATOR_MODEL,
        image_mime=image_mime,
        job_id=job_id,
        output_dir=OUTPUTS_DIR,
    )
    logger.info("[motion] jobId=%s 6frame sheet generated size=%s bytes", job_id, len(sheet_6frame))
    # 로컬 outputs에 저장
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in job_id)
    out_path = OUTPUTS_DIR / f"{safe_name}_6frame.png"
    out_path.write_bytes(sheet_6frame)
    logger.info("[motion] jobId=%s 6frame saved to %s (%s bytes)", job_id, out_path, len(sheet_6frame))

    logger.info("[motion] jobId=%s uploading 6frame PNG to uploadUrl...", job_id)
    r = _put_png_to_url(upload_url, sheet_6frame)
    logger.info("[motion] jobId=%s uploadUrl PUT status=%s", job_id, r.status_code)
    if r.status_code >= 400:
        logger.error("[motion] jobId=%s upload failed response=%s", job_id, r.text or r.reason_phrase)
        raise RuntimeError(f"Storage returned {r.status_code}: {r.text or r.reason_phrase}")


async def _post_callback(
    callback_url: str,
    job_id: str,
    user_id: int,
    success: bool,
    error: str | None = None,
) -> None:
    """[흐름 4] 웹훅 콜백 1회 POST. _run_motion_job에서 성공/실패 시 1회씩 호출."""
    payload = {"jobId": job_id, "userId": user_id, "success": success}
    if not success and error:
        payload["error"] = error[:500]  # 길이 제한
    logger.info("[motion] jobId=%s posting callback success=%s payload=%s", job_id, success, payload)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(callback_url, json=payload)
        logger.info("[motion] jobId=%s callback POST status=%s response_body=%s", job_id, r.status_code, (r.text[:200] if r.text else ""))
    except Exception as e:
        logger.warning("[motion] jobId=%s callback POST failed url=%s err=%s", job_id, callback_url, e)


def _is_503_or_unavailable(exc: BaseException) -> bool:
    """Gemini 503 / UNAVAILABLE / high demand 예외인지 확인 (__cause__ 포함)."""
    msg = (str(exc) or "").lower()
    if "503" in msg or "unavailable" in msg or "high demand" in msg:
        return True
    cause = getattr(exc, "__cause__", None)
    return _is_503_or_unavailable(cause) if cause else False


async def _run_motion_job(body: MotionRequest) -> None:
    """[흐름 2] 백그라운드 태스크. create_motion에서 create_task로 호출. 프로필 GET → _generate_and_upload_motion_sheet(executor) → _post_callback."""
    job_id = body.jobId
    try:
        logger.info("[motion] jobId=%s background job started", job_id)
        logger.info("[motion] jobId=%s fetching profile from profileUrl...", job_id)
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(body.profileUrl)
        logger.info(
            "[motion] jobId=%s profile GET status=%s content_length=%s content_type=%s",
            job_id,
            resp.status_code,
            len(resp.content) if resp.content else 0,
            resp.headers.get("content-type", ""),
        )
        if resp.status_code >= 400 or not resp.content:
            err_msg = f"profile download failed status={resp.status_code}"
            logger.warning("[motion] jobId=%s %s", job_id, err_msg)
            await _post_callback(body.callbackUrl, body.jobId, body.userId, success=False, error=err_msg)
            return
        profile_bytes = resp.content
        image_mime = resp.headers.get("content-type") or "image/png"
        logger.info("[motion] jobId=%s profile ok size=%s bytes, starting 6frame sheet generation...", job_id, len(profile_bytes))

        loop = asyncio.get_event_loop()
        last_exc = None
        for attempt in range(MOTION_503_MAX_RETRIES + 1):
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: _generate_and_upload_motion_sheet(
                            profile_bytes,
                            image_mime,
                            body.uploadUrl,
                            job_id=body.jobId,
                        ),
                    ),
                    timeout=MOTION_JOB_TIMEOUT,
                )
                last_exc = None
                break
            except asyncio.TimeoutError:
                last_exc = None
                raise
            except Exception as e:
                last_exc = e
                if attempt < MOTION_503_MAX_RETRIES and _is_503_or_unavailable(e):
                    delay = MOTION_503_BACKOFF_SECS[min(attempt, len(MOTION_503_BACKOFF_SECS) - 1)]
                    logger.warning(
                        "[motion] jobId=%s 503/UNAVAILABLE, retry %s/%s in %ss: %s",
                        job_id, attempt + 1, MOTION_503_MAX_RETRIES, delay, str(e)[:200],
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

        logger.info("[motion] jobId=%s 6frame sheet generated and uploaded successfully", job_id)
        await _post_callback(body.callbackUrl, body.jobId, body.userId, success=True)
        logger.info("[motion] jobId=%s callback sent success=true, job done", job_id)
    except asyncio.TimeoutError:
        err_msg = f"timed out after {MOTION_JOB_TIMEOUT}s"
        logger.error("[motion] jobId=%s %s", job_id, err_msg)
        await _post_callback(body.callbackUrl, body.jobId, body.userId, success=False, error=err_msg)
        logger.info("[motion] jobId=%s callback sent success=false (timeout), job done", job_id)
    except Exception as e:
        err_msg = str(e)
        logger.exception("[motion] jobId=%s failed: %s", job_id, err_msg)
        await _post_callback(body.callbackUrl, body.jobId, body.userId, success=False, error=err_msg)
        logger.info("[motion] jobId=%s callback sent success=false (error), job done", job_id)


@app.post("/motion")
async def create_motion(body: MotionRequest):
    """
    [흐름 1] 진입점. 요청 수신 즉시 200 OK 반환.
    실제 작업은 _run_motion_job 백그라운드에서 진행하며,
    완료/실패 시 callbackUrl로 POST: { "jobId", "userId", "success": true|false }.
    """
    logger.info(
        "[motion] POST /motion received jobId=%s userId=%s profileUrl=%s uploadUrl=%s callbackUrl=%s",
        body.jobId,
        body.userId,
        body.profileUrl[:80] + "..." if len(body.profileUrl) > 80 else body.profileUrl,
        body.uploadUrl[:80] + "..." if len(body.uploadUrl) > 80 else body.uploadUrl,
        body.callbackUrl[:80] + "..." if len(body.callbackUrl) > 80 else body.callbackUrl,
    )
    if not body.profileUrl.strip():
        logger.warning("[motion] jobId=%s validation failed: profileUrl empty", body.jobId)
        raise HTTPException(400, "profileUrl is required")
    if not body.uploadUrl.strip():
        logger.warning("[motion] jobId=%s validation failed: uploadUrl empty", body.jobId)
        raise HTTPException(400, "uploadUrl is required")
    if not body.callbackUrl.strip():
        logger.warning("[motion] jobId=%s validation failed: callbackUrl empty", body.jobId)
        raise HTTPException(400, "callbackUrl is required")

    asyncio.create_task(_run_motion_job(body))
    logger.info("[motion] jobId=%s task created, returning 200", body.jobId)
    return Response(status_code=200)
