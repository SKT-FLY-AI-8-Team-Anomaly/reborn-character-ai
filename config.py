"""환경 변수 및 설정."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Gemini (이미지 생성 = Nano Banana Pro, 평가 = 텍스트/비전)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GENERATOR_MODEL = "gemini-3-pro-image-preview"  # Nano Banana Pro (최상위 이미지 생성)
DIRECTOR_MODEL = "gemini-2.0-flash"            # 기본 에이전트 (평가용)

# remove.bg API (배경 제거)
REMOVE_BG_API_KEY = os.getenv("REMOVE_BG_API_KEY", "")

# 출력
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
