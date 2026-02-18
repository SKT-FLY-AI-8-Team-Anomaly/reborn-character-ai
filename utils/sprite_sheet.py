"""
6프레임 가로 스프라이트 시트 → 4방향 8프레임 시트 후처리.

- 유령 픽셀 세척 (알파 200 이하 → 완전 투명)
- 여백 크롭 → 6등분 → 좌측 2프레임 반전으로 8프레임 구성
- 프레임별 타이트 크롭 → 격자 배치(발끝 정렬) → PNG bytes 반환
"""
import io
from PIL import Image, ImageOps


def process_6frame_to_4dir_8frame(image_bytes: bytes) -> bytes:
    """
    누끼 딴 6프레임 가로 시트(1x6) 이미지를 받아
    4방향 8프레임 시트(1x8) PNG bytes로 반환.

    입력 프레임 순서 가정: Idle R, Walk R, Idle Front, Walk F, Idle Back, Walk Back
    출력 순서: Left Idle, Left Walk, Right Idle, Right Walk, Front Idle, Front Walk, Back Idle, Back Walk
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    # [핵심 1] 유령 픽셀 세척: 알파 200 이하 → 완전 투명
    pixel_data = img.getdata()
    clean_pixels = []
    for r, g, b, a in pixel_data:
        if a > 200:
            clean_pixels.append((r, g, b, 255))
        else:
            clean_pixels.append((0, 0, 0, 0))
    img.putdata(clean_pixels)

    # 전체 여백 1차 컷
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    # 6등분
    num_original_frames = 6
    frame_w = img.width // num_original_frames
    frame_h = img.height
    original_frames = [
        img.crop((i * frame_w, 0, (i + 1) * frame_w, frame_h))
        for i in range(num_original_frames)
    ]

    # 왼쪽 방향: Right 프레임(0, 1) 좌우 반전
    left_idle = ImageOps.mirror(original_frames[0])
    left_walk = ImageOps.mirror(original_frames[1])

    # 8프레임 순서: Left Idle, Left Walk, Right Idle, Right Walk, Front Idle, Front Walk, Back Idle, Back Walk
    final_frames_list = [
        left_idle,
        left_walk,
        original_frames[0],
        original_frames[1],
        original_frames[2],
        original_frames[3],
        original_frames[4],
        original_frames[5],
    ]

    # 각 프레임별 타이트한 알맹이(Bounding Box) 추출
    trimmed_frames = []
    max_w, max_h = 0, 0
    for frame in final_frames_list:
        b = frame.getbbox()
        if b:
            t = frame.crop(b)
            trimmed_frames.append(t)
            max_w = max(max_w, t.width)
            max_h = max(max_h, t.height)
        else:
            trimmed_frames.append(frame)
            max_w = max(max_w, frame.width)
            max_h = max(max_h, frame.height)

    # 격자 배치 (여유 공간 20px)
    grid_size = max(max_w, max_h) + 20
    final_sheet = Image.new("RGBA", (grid_size * 8, grid_size), (0, 0, 0, 0))

    for i, trimmed in enumerate(trimmed_frames):
        x_off = (grid_size - trimmed.width) // 2
        y_off = grid_size - trimmed.height  # [핵심 2] 발끝 정렬로 널뛰기 방지
        final_sheet.paste(trimmed, (i * grid_size + x_off, y_off))

    buf = io.BytesIO()
    final_sheet.save(buf, format="PNG")
    return buf.getvalue()


# 8프레임 시트에서 사용하는 프레임 순서 (process_6frame_to_4dir_8frame 출력과 동일)
FRAME_ORDER_8 = (
    "left_idle",
    "left_walk",
    "right_idle",
    "right_walk",
    "front_idle",
    "front_walk",
    "back_idle",
    "back_walk",
)


def slice_8frame_sheet_to_frames(sheet_bytes: bytes) -> list[bytes]:
    """
    1x8 스프라이트 시트 PNG를 8장의 PNG bytes로 자른다.
    반환 순서: left_idle, left_walk, right_idle, right_walk, front_idle, front_walk, back_idle, back_walk
    """
    img = Image.open(io.BytesIO(sheet_bytes)).convert("RGBA")
    w, h = img.size
    cell_w = w // 8
    frames = []
    for i in range(8):
        box = (i * cell_w, 0, (i + 1) * cell_w, h)
        cell = img.crop(box)
        buf = io.BytesIO()
        cell.save(buf, format="PNG")
        frames.append(buf.getvalue())
    return frames
