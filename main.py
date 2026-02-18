"""
Pixel Refiner: 사진 + 지정 프롬프트로 1장 생성 → 그 이미지에 대해 디렉터와 피드백 주고받기.

사용법: python main.py <사진 경로> [--max-iteration N] [--output 경로]
버전 관리: outputs/<stem>_<timestamp>/ 에 iter_0.png, iter_1.png, ..., history.json 저장.
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from google.genai.errors import ClientError

from config import GEMINI_API_KEY, GENERATOR_MODEL, DIRECTOR_MODEL, OUTPUT_DIR
from agents.generator_agent import generate_initial, refine
from agents.director_agent import evaluate

# 429 재시도: 최대 시도 횟수, 초기 대기(초)
MAX_RETRIES = 5
RETRY_INITIAL_WAIT = 10


def _is_rate_limit(err: BaseException) -> bool:
    if isinstance(err, ClientError):
        return getattr(err, "code", None) == 429
    return False


def _call_with_retry(fn, *args, **kwargs):
    """429 발생 시 지수 백오프로 재시도."""
    last_err = None
    wait = RETRY_INITIAL_WAIT
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            if _is_rate_limit(e) and attempt < MAX_RETRIES - 1:
                print(f"  [429 rate limit] Waiting {wait}s before retry ({attempt + 1}/{MAX_RETRIES})...")
                time.sleep(wait)
                wait = min(wait * 2, 120)
            else:
                raise
    raise last_err


def run_pixel_refiner(
    photo_path: str | Path,
    max_iteration: int = 5,
    output_path: str | Path | None = None,
    api_key: str | None = None,
) -> tuple[bytes, dict]:
    """
    사진 + 지정 프롬프트로 1장 생성 → 그 이미지에 대해 디렉터 평가·수정 반복.
    Returns: (final_sprite_bytes, last_evaluation_dict)
    """
    api_key = api_key or GEMINI_API_KEY
    photo_path = Path(photo_path)
    if not photo_path.is_file():
        raise FileNotFoundError(f"Photo not found: {photo_path}")

    out_dir = Path(output_path).parent if output_path else OUTPUT_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sprite_{photo_path.stem}"
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"{stem}_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    sprite_bytes = _call_with_retry(
        lambda: generate_initial(photo_path, api_key=api_key, model=GENERATOR_MODEL)
    )
    (run_dir / "iter_0.png").write_bytes(sprite_bytes)
    history: list[dict] = [
        {
            "iteration": 0,
            "image_file": "iter_0.png",
            "refinement_prompt": None,
            "evaluation": None,
        }
    ]
    last_eval = None

    for iteration in range(1, max_iteration + 1):
        last_eval = _call_with_retry(
            lambda: evaluate(sprite_bytes, api_key=api_key, model=DIRECTOR_MODEL)
        )
        # 방향별 피드백에서 critique/instruction 조합
        directional = last_eval.get("directional_feedback") or {}
        retry_directions = []
        instruction_parts = []
        for direction, label in [("down", "Row 1 Down"), ("left", "Row 2 Left"), ("right", "Row 3 Right"), ("up", "Row 4 Up")]:
            fb = directional.get(direction) or {}
            if fb.get("status") == "RETRY":
                if fb.get("actionable_instruction", "").strip():
                    instruction_parts.append(f"{label}: {fb['actionable_instruction'].strip()}")
                retry_directions.append(direction)
        critique = f"RETRY rows: {', '.join(retry_directions)}" if retry_directions else ""
        instruction = "\n".join(instruction_parts) if instruction_parts else ""

        history[-1]["evaluation"] = {
            "overall_score": last_eval.get("overall_score"),
            "overall_status": last_eval.get("overall_status"),
            "directional_feedback": directional,
        }
        print(f"[Iteration {iteration}/{max_iteration}] score={last_eval.get('overall_score')} status={last_eval.get('overall_status')}")
        if critique:
            print(f"  critique: {critique[:120]}...")
        if last_eval.get("overall_status") == "PASS":
            print("  → PASS. Done.")
            break
        if not instruction:
            print("  → No actionable_instruction; stopping.")
            break
        print(f"  → Refining: {instruction[:80]}...")
        print("[피드백→생성] 디렉터 actionable_instruction (전체):")
        print("  ---")
        print(f"  {instruction}")
        print("  ---")
        sprite_bytes = _call_with_retry(
            lambda: refine(
                sprite_bytes,
                instruction,
                api_key=api_key,
                model=GENERATOR_MODEL,
                critique=critique,
            )
        )
        iter_file = f"iter_{iteration}.png"
        (run_dir / iter_file).write_bytes(sprite_bytes)
        history.append(
            {
                "iteration": iteration,
                "image_file": iter_file,
                "refinement_prompt": instruction,
                "evaluation": None,
            }
        )
        if iteration < max_iteration:
            time.sleep(2)

    (run_dir / "final.png").write_bytes(sprite_bytes)
    run_metadata = {
        "timestamp": run_ts,
        "photo_path": str(photo_path),
        "max_iteration": max_iteration,
    }
    history_data = {"run_metadata": run_metadata, "versions": history}
    (run_dir / "history.json").write_text(
        json.dumps(history_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Run saved: {run_dir}")
    print(f"  images: iter_0.png ... iter_{len(history)-1}.png, final.png")
    print(f"  feedback: history.json")

    if output_path:
        Path(output_path).write_bytes(sprite_bytes)
        print(f"Also saved: {output_path}")

    return sprite_bytes, (last_eval or {})


def main():
    parser = argparse.ArgumentParser(description="Pixel Refiner: 사진 → 4프레임 걷기 스프라이트 시트")
    parser.add_argument("photo", type=str, help="참조용 사용자 사진 경로")
    parser.add_argument("--max-iteration", "-n", type=int, default=5, help="생성↔디렉터 최대 반복 횟수 (기본 5)")
    parser.add_argument("--output", "-o", type=str, default=None, help="출력 PNG 경로")
    args = parser.parse_args()

    run_pixel_refiner(
        photo_path=args.photo,
        max_iteration=args.max_iteration,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
