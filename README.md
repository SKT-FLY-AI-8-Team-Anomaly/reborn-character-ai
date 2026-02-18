# Pixel Refiner

사용자 **사진 한 장**을 넣으면, **생성 에이전트**(Nano Banana)와 **피드백 에이전트**(Technical Art Director)가 서로 대화하며, 최종적으로 **4프레임 걷기 2D 도트 스프라이트 시트**를 만들어 줍니다.

## 구조

- **생성 에이전트**: JRPG 'To the Moon' 스타일 픽셀 아트 애니메이터. 모델은 **Nano Banana** (`gemini-2.5-flash-image`).
- **피드백 에이전트**: 물리/해부학 기준으로 스프라이트를 평가하는 Technical Art Director. **기본 에이전트** (`gemini-2.0-flash`) 사용.
- 두 에이전트가 **max_iteration** 횟수만큼 주고받으며, 디렉터가 PASS를 줄 때까지 생성 → 평가 → 수정 반복.

## 설정

1. `.env`에 Gemini API 키 설정:
   ```
   GEMINI_API_KEY=your_key_here
   ```
2. 의존성 설치:
   ```bash
   pip install -r requirements.txt
   ```

## 사용법

### CLI (기존)

```bash
# 사진 + 지정 프롬프트로 1장 생성 → 그 이미지에 대해 디렉터와 피드백 반복
python main.py path/to/photo.jpg

# 반복 횟수 / 출력 경로
python main.py path/to/photo.jpg --max-iteration 3 --output my_sprite.png
```

### API 서버 (FastAPI)

두 단계 API: **사진 → AI 프로필** (`/profile`), **AI 프로필 → 4모션 스프라이트** (`/sprites`).  
(실제 생성 로직 연동은 추후 진행)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

- 서버: http://127.0.0.1:8000  
- API 문서: http://127.0.0.1:8000/docs  
- `POST /profile`: 이미지 파일 업로드 → AI 프로필 생성 (스켈레톤)  
- `POST /sprites`: JSON `{ "profile_id": "..." }` → 4방향 스프라이트 생성 (스켈레톤)

## 출력 (버전 관리)

매 실행마다 `outputs/<sprite_사진이름>_<YYYYMMDD_HHMMSS>/` 폴더가 생성됩니다.

- **iter_0.png** — 최초 생성 이미지
- **iter_1.png, iter_2.png, ...** — 디렉터 피드백으로 수정한 이미지 (반복마다 1장)
- **final.png** — 최종 결과 (iter_N과 동일)
- **history.json** — 각 버전별 기록:
  - `run_metadata`: 실행 옵션(사진 경로, max_iteration, three_sheets 등)
  - `versions`: 각 iteration별 `image_file`, `refinement_prompt`(수정에 쓴 디렉터 지시), `evaluation`(점수·status·critique·actionable_instruction)

콘솔에는 매 반복별 디렉터 점수·상태·크리티크·수정 지시가 출력됩니다.
