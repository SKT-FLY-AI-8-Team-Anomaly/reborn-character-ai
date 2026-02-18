"""
Pixel Art 생성 에이전트.
사진 + 지정 프롬프트로 1장 생성 → 디렉터 피드백으로 같은 이미지만 반복 수정.
"""
import io
from pathlib import Path
from PIL import Image

from google import genai
from google.genai import types

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
