"""
Pixel Art 생성 에이전트.
사진 + 지정 프롬프트로 1장 생성 → 디렉터 피드백으로 같은 이미지만 반복 수정.
피드백 에이전트(Evaluator)로 6프레임 시트 검수 후 실패 시 refine_prompt로 재시도.
"""
import io
import json
import logging
from pathlib import Path
from PIL import Image

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# 맨 처음 생성 시 사용하는 프롬프트 (그대로 한 번에 전달)
INITIAL_PROMPT = """Using the uploaded real photo as a reference, create a 2D pixel art character sprite sheet representing the person at their current age.

The output should be a 4x4 grid PNG sprite sheet (4 rows arranged vertically, 4 frames per row horizontally) with a transparent background. The sheet must cover 4 directions (Down, Left, Right, Up) in JRPG style (similar to 'To the Moon'), with a consistent resolution (around 32x48 pixels per single frame). Row 1 (Top): 4-frame smooth Front-facing (Walking Down) animation cycle. Row 2: 4-frame smooth Left-facing (Walking Left) animation cycle. Row 3: 4-frame smooth Right-facing (Walking Right) animation cycle. Row 4 (Bottom): 4-frame smooth Back-facing (Walking Up) animation cycle.

Keep the face shape, eyes, nose, mouth, hairstyle, clothing, and overall atmosphere as close as possible to the person in the uploaded photo. Do not change the age; maintain the exact look of the reference photo across all frames.

Generate clean pixel art with clear outlines and simple shading, suitable for use as a character sprite in a 2D game engine (like Phaser) with frameWidth and frameHeight equal to each single frame."""

# 프로필 1장 생성용 (To the Moon 스타일, 정면 전신 정지 스프라이트)
PROFILE_PROMPT = """[Task]
Generate a single, static 2D pixel art character sprite based on the provided reference photo.
[Style Guidelines - 'To The Moon' Aesthetic]

Genre: Narrative JRPG pixel art (e.g., To the Moon, SNES-era RPGs).
Resolution: Low resolution, suitable for a single game character asset (approximately 48 to 64 pixels tall).
Technique: Clean dark outlines (black or very dark color), simple 2-3 tone cell shading (base color + shadow), no complex gradients or heavy dithering on the character itself.
Background: Completely transparent PNG.
[CRITICAL REFERENCE INSTRUCTIONS - Face & Likeness]

Primary Goal: Create a full-body character where the head is the primary focus of accuracy. You must salvage the exact facial features from the reference photo and translate them into pixel art with maximum fidelity possible at this resolution.
Face Detail: Capture the specific face shape, hairstyle, eye color/shape, nose structure, and mouth expression from the photo. Despite the low pixel count, the character needs to be instantly recognizable as the specific person in the image.
Age: Maintain the exact current age of the person in the photo. Do not make them look younger, older, or "chibi" style.
[Body & Clothing Instructions]

Pose: Static, full-body, standing, facing straight front.
Clothing: accurately translate the clothing visible in the reference photo into pixel form, paying close attention to colors, patterns, and clothing type. If only the top half is visible in the photo, extrapolate the bottom half logically to match the style of the top (e.g., if wearing a business shirt, add matching trousers).
[Exclusions]

Do NOT generate a sprite sheet, grid, walking animation frames, or a background scene. Only a single, isolated standing character."""


# 4방향 모션: 프로필 이미지 → front/back/left/right 각 1장 (정지 픽셀 아트)
MOTION_DIRECTION_PROMPT = """Using the provided character profile image as the only reference, generate a single static 2D pixel art image of the same character from the {direction} view.

Requirements:
- Style: Same as To the Moon / JRPG (clean dark outlines, 2-3 tone cell shading, transparent background).
- Resolution: Approximately 48 to 64 pixels tall, single character only.
- Output: Exactly ONE image — the character from the {direction} view. No grid, no sprite sheet, no animation frames.
- Consistency: Keep face, hair, clothing colors and style identical to the reference; only the viewing angle is {direction}."""

DIRECTION_NAMES = ("front", "back", "left", "right")


# 6프레임 가로 1열 스프라이트 시트 (1x6, SNES JRPG 스타일) — 입력은 프로필/캐릭터 이미지 1장
SIX_FRAME_HORIZONTAL_SHEET_PROMPT = """Create ONE single large horizontal sprite sheet image on a transparent background.

This must be ONE unified image file.
NOT multiple images.
NOT separate files.
NOT a grid layout.
NOT stacked vertically.

The canvas must contain EXACTLY 6 sprites arranged in ONE strict horizontal row (1x6 layout).
All sprites must be aligned in a single straight line from left to right.

The image must be extremely wide horizontally (width ≈ 6 times the height).
No wrapping.
No 2x3 layout.
No vertical stacking.

Each sprite must:
- Have identical width and height
- Be evenly spaced
- Not overlap
- Not be cropped
- Be centered inside its frame space

All six sprites must strictly maintain the 16-bit SNES JRPG pixel art style, color palette, pixel density, shading logic, and character proportions of the provided character reference image.

Perfect consistency is required across all frames:
- same scale
- same proportions
- same colors
- same lighting
Only pose and direction may change.

--------------------------------
FRAME ORDER (Left to Right)
--------------------------------

Frame 1 — Idle Right:
Complete right-facing profile. Standing still. Feet together flat on the ground. Arms relaxed. Clear side profile.

Frame 2 — Walking Right:
Right-facing profile walking to the right. Mid-stride. Right leg forward, left leg back. Opposite arms swinging naturally.

Frame 3 — Idle Front:
Static front-facing pose identical to the provided character reference image. Feet together. Arms relaxed. Looking forward.

Frame 4 — Walking Forward:
Front-facing walking toward the viewer. One foot forward (slightly larger or lower to indicate depth). Opposite arms swinging. Torso remains front-facing.

Frame 5 — Idle Back:
Fully back-facing. Standing still. Feet together. Arms relaxed. No front emblem visible.

Frame 6 — Walking Backward:
Back-facing walking away from the viewer. One foot stepping away (slightly higher/smaller to indicate depth). Opposite arms swinging. Torso remains facing away.

--------------------------------
TECHNICAL RULES
--------------------------------

True transparent background (alpha channel).
No gradient.
No environment.
No extra characters.
No camera rotation.
No diagonal composition.
Small pixel shadows only beneath walking poses."""


# 피드백 에이전트(Evaluator): 생성된 6프레임 시트를 원본 프로필과 비교해 검수, JSON(status, reason, refine_prompt) 반환
EVALUATOR_SYSTEM_PROMPT = """You are an uncompromising QA Evaluator and Art Director for 2D game assets.
Your task is to strictly evaluate a generated sprite sheet against the original character design (the FIRST image) and the required generation rules. The SECOND image is the generated sprite sheet to evaluate.

Evaluate the generated image using the following strict checklist. You MUST check them in this exact order of priority. Stop at the FIRST failure you find. Focus only on the highest-priority failure.

[PRIORITY 1: LAYOUT & COUNT]
- Count the sprites. Are there EXACTLY 6 sprites?
- Is the layout a single, extremely wide horizontal panorama?
- Are the 6 sprites arranged side-by-side in ONE single straight horizontal line?

[PRIORITY 2: POSES & SEQUENCE (Left to Right)]
Check the strict 6-frame sequence:
- Frame 1 (Idle Right): Right-facing profile, standing still.
- Frame 2 (Walk Right): Right-facing profile, actively walking.
- Frame 3 (Idle Front): Front-facing, standing still.
- Frame 4 (Walk Front): Front-facing, actively walking.
- Frame 5 (Idle Back): FULLY back-facing, standing still.
- Frame 6 (Walk Back): FULLY back-facing, actively walking.

[PRIORITY 3: STYLE & CONSISTENCY]
- Does the generated character strictly maintain the exact same hair color, clothing design, and proportions as the first image (reference)?
- Is it strictly 16-bit SNES JRPG pixel art?

[OUTPUT FORMAT & FEEDBACK RULES]
You MUST respond ONLY in valid JSON format with exactly these keys: "status", "reason", "refine_prompt".

If ANY check fails, you must write a "refine_prompt". Crucial Rule for refine_prompt: Use ONLY POSITIVE VISUAL REINFORCEMENT. Image generation AI ignores negative words (like "not", "no", "never").
- Instead of saying "do not make a grid", command it to "GENERATE AN EXTREMELY WIDE 6:1 PANORAMIC BANNER".
- Instead of saying "not 8 sprites", command it to "GENERATE EXACTLY 6 SPRITES ONLY".

[JSON STRUCTURE]
- If ALL checks pass perfectly: {"status": "PASS", "reason": "All criteria met.", "refine_prompt": ""}
- If ANY check fails: {"status": "FAIL", "reason": "[Brief failure reason]", "refine_prompt": "CRITICAL FIX REQUIRED: [Strong, positive visual instruction]"}
"""


# 수정 시: 문제(critique) + 지시(actionable) 전달해 맥락을 줌
REFINE_PREFIX = """The current sprite sheet image has defects. You MUST output a modified image that fixes them. Keep the character's face, hair, clothes, and colors identical; only fix the animation/anatomy as instructed.

"""


def _ensure_client(api_key: str):
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in environment.")
    return genai.Client(api_key=api_key)


def _extract_image_from_response(response) -> bytes | None:
    """GenerateContent 응답에서 이미지 바이너리 추출."""
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None and part.inline_data.data:
            return part.inline_data.data
    return None


def generate_profile(
    image_bytes: bytes,
    api_key: str,
    model: str = "gemini-3-pro-image-preview",
    image_mime: str = "image/png",
) -> bytes:
    """원본 이미지 바이트를 받아 캐릭터 프로필 1장(PNG) 생성. 배경/포즈 정규화."""
    client = _ensure_client(api_key)
    if not image_mime or not image_mime.startswith("image/"):
        image_mime = "image/png"
    contents = [
        types.Part.from_bytes(data=image_bytes, mime_type=image_mime),
        types.Part.from_text(text=PROFILE_PROMPT),
    ]
    config = types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    out = _extract_image_from_response(response)
    if out is None:
        raise RuntimeError("Generator did not return an image for profile.")
    return out


def generate_direction_view(
    profile_image_bytes: bytes,
    direction: str,
    api_key: str,
    model: str = "gemini-3-pro-image-preview",
    image_mime: str = "image/png",
) -> bytes:
    """프로필 이미지 1장을 입력으로, 지정 방향(front/back/left/right) 뷰 1장 생성."""
    if direction not in DIRECTION_NAMES:
        raise ValueError(f"direction must be one of {DIRECTION_NAMES}, got {direction!r}")
    client = _ensure_client(api_key)
    if not image_mime or not image_mime.startswith("image/"):
        image_mime = "image/png"
    prompt = MOTION_DIRECTION_PROMPT.format(direction=direction)
    contents = [
        types.Part.from_bytes(data=profile_image_bytes, mime_type=image_mime),
        types.Part.from_text(text=prompt),
    ]
    config = types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    out = _extract_image_from_response(response)
    if out is None:
        raise RuntimeError(f"Generator did not return an image for direction {direction}.")
    return out


def generate_four_directions(
    profile_image_bytes: bytes,
    api_key: str,
    model: str = "gemini-3-pro-image-preview",
    image_mime: str = "image/png",
) -> dict[str, bytes]:
    """프로필 이미지 → front, back, left, right 각 1장. 반환값: { 'front': bytes, ... }."""
    result: dict[str, bytes] = {}
    for direction in DIRECTION_NAMES:
        result[direction] = generate_direction_view(
            profile_image_bytes, direction, api_key, model, image_mime
        )
    return result


def generate_six_frame_horizontal_sheet(
    profile_image_bytes: bytes,
    api_key: str,
    model: str = "gemini-3-pro-image-preview",
    image_mime: str = "image/png",
    extra_prompt: str | None = None,
) -> bytes:
    """프로필 이미지 1장을 입력으로, 1x6 가로 스프라이트 시트 1장 생성. extra_prompt가 있으면 기본 프롬프트 뒤에 붙여 재시도용 지시 반영."""
    client = _ensure_client(api_key)
    if not image_mime or not image_mime.startswith("image/"):
        image_mime = "image/png"
    prompt = SIX_FRAME_HORIZONTAL_SHEET_PROMPT
    if extra_prompt and extra_prompt.strip():
        prompt = prompt + "\n\n" + extra_prompt.strip()
    contents = [
        types.Part.from_bytes(data=profile_image_bytes, mime_type=image_mime),
        types.Part.from_text(text=prompt),
    ]
    config = types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    out = _extract_image_from_response(response)
    if out is None:
        raise RuntimeError("Generator did not return an image for 6-frame sheet.")
    return out


def remove_background(image_bytes: bytes, api_key: str | None = None) -> bytes:
    """이미지 바이트에 대해 배경 제거(누끼) 후 PNG bytes 반환. remove.bg API 사용."""
    import requests
    from config import REMOVE_BG_API_KEY

    key = api_key or REMOVE_BG_API_KEY
    if not key or not key.strip():
        raise ValueError("REMOVE_BG_API_KEY is required for background removal. Set it in .env")

    response = requests.post(
        "https://api.remove.bg/v1.0/removebg",
        files={"image_file": ("image.png", io.BytesIO(image_bytes), "image/png")},
        data={"size": "auto"},
        headers={"X-Api-Key": key},
        timeout=120,
    )
    if response.status_code != requests.codes.ok:
        raise RuntimeError(
            f"remove.bg API error: {response.status_code} {response.text or response.reason_phrase}"
        )
    return response.content


def evaluate_sprite_sheet(
    profile_image_bytes: bytes,
    generated_sheet_bytes: bytes,
    api_key: str,
    model: str = "gemini-2.0-flash",
) -> dict:
    """
    피드백 에이전트: 원본 프로필(1장) + 생성된 6프레임 시트(2장)를 보내고,
    JSON { status: "PASS"|"FAIL", reason: str, refine_prompt: str } 반환.
    """
    from config import DIRECTOR_MODEL
    client = _ensure_client(api_key)
    eval_model = model or DIRECTOR_MODEL
    contents = [
        types.Part.from_bytes(data=profile_image_bytes, mime_type="image/png"),
        types.Part.from_bytes(data=generated_sheet_bytes, mime_type="image/png"),
        types.Part.from_text(
            "Evaluate the second image (generated sprite sheet) against the first (reference character). Return ONLY valid JSON with keys: status, reason, refine_prompt."
        ),
    ]
    config = types.GenerateContentConfig(
        system_instruction=EVALUATOR_SYSTEM_PROMPT,
        response_mime_type="application/json",
        temperature=0.1,
    )
    response = client.models.generate_content(
        model=eval_model,
        contents=contents,
        config=config,
    )
    text = getattr(response, "text", None) or ""
    if not text:
        raise RuntimeError("Evaluator did not return any text.")
    out = json.loads(text)
    if not isinstance(out, dict) or "status" not in out:
        raise RuntimeError(f"Evaluator returned invalid JSON: {text[:200]}")
    return out


def generate_six_frame_sheet_with_rembg(
    profile_image_bytes: bytes,
    api_key: str,
    model: str = "gemini-3-pro-image-preview",
    image_mime: str = "image/png",
    evaluator_model: str = "gemini-2.0-flash",
    max_eval_attempts: int = 3,
    job_id: str | None = None,
    output_dir: Path | str | None = None,
) -> bytes:
    """
    6프레임 가로 시트 생성 → 피드백 에이전트로 검수 → FAIL 시 refine_prompt로 재생성(최대 max_eval_attempts회)
    → PASS 시 remove.bg로 배경 제거 후 PNG bytes 반환.
    job_id가 있으면 output_dir에 attempt별 이미지·피드백 JSON 저장.
    """
    from config import DIRECTOR_MODEL, OUTPUT_DIR
    eval_model = evaluator_model or DIRECTOR_MODEL
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    save_refine_artifacts = bool(job_id and out_dir)
    safe_prefix = "".join(c if c.isalnum() or c in "-_" else "_" for c in job_id) if job_id else "refine"

    sheet = None
    extra_prompt = None
    for attempt in range(max_eval_attempts):
        attempt_num = attempt + 1
        logger.info("[evaluator] attempt %s/%s — generating 6frame sheet (extra_prompt=%s)", attempt_num, max_eval_attempts, "yes" if extra_prompt else "no")
        if extra_prompt:
            logger.info("[evaluator] refine_prompt applied (length=%s): %s", len(extra_prompt), extra_prompt[:120] + "..." if len(extra_prompt) > 120 else extra_prompt)

        sheet = generate_six_frame_horizontal_sheet(
            profile_image_bytes, api_key, model, image_mime, extra_prompt=extra_prompt
        )
        logger.info("[evaluator] attempt %s — 6frame sheet generated size=%s bytes", attempt_num, len(sheet))

        if save_refine_artifacts:
            out_dir.mkdir(parents=True, exist_ok=True)
            img_path = out_dir / f"{safe_prefix}_attempt_{attempt_num}.png"
            img_path.write_bytes(sheet)
            logger.info("[evaluator] attempt %s — saved image to %s", attempt_num, img_path)

        logger.info("[evaluator] attempt %s — calling feedback agent (profile + generated image)...", attempt_num)
        result = evaluate_sprite_sheet(
            profile_image_bytes, sheet, api_key, model=eval_model
        )
        status = (result.get("status") or "").strip().upper()
        reason = result.get("reason") or ""
        refine_prompt = (result.get("refine_prompt") or "").strip()

        logger.info("[evaluator] attempt %s — feedback status=%s reason=%s", attempt_num, status, reason)
        logger.info("[evaluator] attempt %s — refine_prompt (%s chars): %s", attempt_num, len(refine_prompt), refine_prompt[:150] + "..." if len(refine_prompt) > 150 else refine_prompt)

        if save_refine_artifacts:
            feedback_path = out_dir / f"{safe_prefix}_feedback_{attempt_num}.json"
            feedback_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("[evaluator] attempt %s — saved feedback to %s", attempt_num, feedback_path)

        if status == "PASS":
            logger.info("[evaluator] attempt %s — PASS, stopping refine loop, proceeding to remove.bg", attempt_num)
            break
        if not refine_prompt:
            logger.warning("[evaluator] attempt %s — FAIL but no refine_prompt, stopping retries", attempt_num)
            break
        extra_prompt = refine_prompt
        logger.info("[evaluator] attempt %s — FAIL, will retry with refine_prompt in next attempt", attempt_num)

    if sheet is None:
        raise RuntimeError("6-frame sheet generation did not produce an image.")
    logger.info("[evaluator] calling remove.bg on final sheet...")
    return remove_background(sheet)


def generate_four_dir_8frame_sheet(
    profile_image_bytes: bytes,
    api_key: str,
    model: str = "gemini-3-pro-image-preview",
    image_mime: str = "image/png",
) -> bytes:
    """
    6프레임 가로 시트 생성 → remove.bg API 누끼 → 4방향 8프레임 후처리까지 한 번에 수행.
    반환: 1x8 격자 시트 PNG bytes (Left Idle/Walk, Right Idle/Walk, Front Idle/Walk, Back Idle/Walk).
    """
    from utils.sprite_sheet import process_6frame_to_4dir_8frame

    sheet_no_bg = generate_six_frame_sheet_with_rembg(
        profile_image_bytes, api_key, model, image_mime
    )
    return process_6frame_to_4dir_8frame(sheet_no_bg)


def generate_initial(
    photo_path: str | Path,
    api_key: str,
    model: str = "gemini-3-pro-image-preview",
) -> bytes:
    """사진 + INITIAL_PROMPT 그대로 한 번에 전달해서 1장 생성."""
    client = _ensure_client(api_key)
    path = Path(photo_path)
    if not path.is_file():
        raise FileNotFoundError(f"Photo not found: {photo_path}")

    image = Image.open(path)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    contents = [
        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        types.Part.from_text(text=INITIAL_PROMPT),
    ]
    config = types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    out = _extract_image_from_response(response)
    if out is None:
        raise RuntimeError("Generator did not return an image. Check model and prompt.")
    return out


def refine(
    sprite_image_bytes: bytes,
    correction_instruction: str,
    api_key: str,
    model: str = "gemini-3-pro-image-preview",
    critique: str | None = None,
) -> bytes:
    """디렉터의 critique(문제) + actionable_instruction(지시)를 넘겨서 수정."""
    client = _ensure_client(api_key)
    body = REFINE_PREFIX
    if critique and critique.strip():
        body += f"**Problems in the current image:** {critique.strip()}\n\n**Correction to apply:** {correction_instruction.strip()}"
    else:
        body += f"**Correction to apply:** {correction_instruction.strip()}"
    print("[생성 에이전트] refine 호출 시 API에 들어가는 전체 프롬프트:")
    print("  ---")
    print(body)
    print("  ---")

    contents = [
        types.Part.from_bytes(data=sprite_image_bytes, mime_type="image/png"),
        types.Part.from_text(text=body),
    ]
    config = types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    out = _extract_image_from_response(response)
    if out is None:
        raise RuntimeError("Generator did not return an image on refinement.")
    return out
