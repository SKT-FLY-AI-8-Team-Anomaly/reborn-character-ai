"""
Technical Art Director 에이전트. 4x4 스프라이트 시트를 4가지 규칙으로 평가하고
방향별(down/left/right/up) 피드백 JSON 반환.
"""
import json
from google import genai
from google.genai import types

DIRECTOR_SYSTEM = """You are a strict Technical Art Director. Your job is to approve or reject pixel art animations based on physics and anatomy.

### Evaluation Checklist (The "4 Commandments")
Analyze the given 4x4 sprite sheet against these 4 rules. Be extremely pedantic.

1. **Leg Crossing (Mechanics):**
   - In side-walking frames (Left/Right), legs MUST cross.
   - Failure: Walking looks like sliding (Moonwalking).
   - Pass: One frame clearly shows Left Leg forward, another shows Right Leg forward.

2. **Vertical Bounce (Physics):**
   - Track the top-most pixel of the head across the 4 frames of a row.
   - There MUST be a vertical difference (1-2px) between the 'Contact' pose (lowest) and 'Passing' pose (highest).
   - Failure: Head stays at the exact same Y-level (Floating).

3. **Cranial Volume (Anatomy):**
   - In Side View, the head depth (front-to-back) must match the Front View width.
   - Failure: Side profile looks flat or crushed at the back.

4. **Pixel Width Consistency (Tech):**
   - Measure character width in Front View vs. Side/Back View.
   - They MUST be identical."""

# 프롬프트를 수정하여 4방향 각각에 대한 피드백을 요구합니다.
EVALUATION_PROMPT = """Analyze this 4x4 sprite sheet.
Row 1: Down (Front)
Row 2: Left
Row 3: Right
Row 4: Up (Back)

Evaluate EACH row individually against the 4 Commandments. Provide a JSON response matching this exact structure:

{
    "overall_score": <integer 0-100>,
    "overall_status": "PASS" or "RETRY",
    "directional_feedback": {
        "down": {
            "status": "PASS" or "RETRY",
            "actionable_instruction": "Leave empty if PASS. If RETRY, give ONE specific pixel-level instruction for Row 1."
        },
        "left": {
            "status": "PASS" or "RETRY",
            "actionable_instruction": "Leave empty if PASS. If RETRY, give ONE specific instruction for Row 2. e.g., 'Make legs cross in frames 1 and 3.'"
        },
        "right": {
            "status": "PASS" or "RETRY",
            "actionable_instruction": "Leave empty if PASS. If RETRY, give ONE specific instruction for Row 3."
        },
        "up": {
            "status": "PASS" or "RETRY",
            "actionable_instruction": "Leave empty if PASS. If RETRY, give ONE specific instruction for Row 4."
        }
    }
}"""


def _ensure_client(api_key: str):
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in environment.")
    return genai.Client(api_key=api_key)


def evaluate(sprite_image_bytes: bytes, api_key: str, model: str = "gemini-2.0-flash") -> dict:
    """
    스프라이트 시트 이미지를 평가해 방향별 피드백이 담긴 JSON 반환.
    """
    client = _ensure_client(api_key)

    contents = [
        types.Part.from_bytes(data=sprite_image_bytes, mime_type="image/png"),
        types.Part.from_text(text=EVALUATION_PROMPT),
    ]

    # JSON 구조를 강제하기 위해 response_mime_type 사용
    config = types.GenerateContentConfig(
        system_instruction=DIRECTOR_SYSTEM,
        response_mime_type="application/json",
        temperature=0.1,  # 일관된 평가를 위해 온도를 낮춤
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    text = getattr(response, "text", None) or ""

    if not text:
        raise RuntimeError("Director did not return any text.")

    return json.loads(text)
