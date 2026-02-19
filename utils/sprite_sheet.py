"""
6프레임 가로 스프라이트 시트 → 4방향 8프레임 시트 후처리.

- 유령 픽셀 세척 (알파 200 이하 → 완전 투명)
- 스마트 슬라이싱: 알파 채널로 캐릭터 경계 감지 후 6프레임 분리 (균등 6등분 대신)
- 좌측 2프레임 반전 → 8프레임 구성 → 격자 배치(발끝 정렬) → PNG bytes 반환
"""
import io
import numpy as np
from PIL import Image, ImageOps


def _equal_slice_6_frames(transparent_image: Image.Image) -> list[Image.Image]:
    """이미지 너비를 균등 6등분하여 6프레임으로 자른다. 스마트 슬라이스 실패 시 폴백용."""
    w, h = transparent_image.size
    if w < 6:
        raise ValueError("SMART_SLICE_ERROR: Image too narrow for 6 frames.")
    step = w // 6
    frames = []
    for i in range(6):
        x0, x1 = i * step, (i + 1) * step if i < 5 else w
        frames.append(transparent_image.crop((x0, 0, x1, h)))
    return frames


def smart_slice_6_frames(transparent_image: Image.Image) -> list[Image.Image]:
    """
    투명 배경의 1x6 이미지를 입력받아, 캐릭터 사이의 빈 공간을 감지하여
    정확히 6개의 개별 캐릭터 이미지 리스트로 분리하여 반환합니다.
    6개를 찾지 못하면 균등 6등분 폴백을 사용합니다.
    """
    img_arr = np.array(transparent_image)
    alpha_channel = img_arr[:, :, 3]

    non_empty_columns = np.where(np.any(alpha_channel > 0, axis=0))[0]
    if len(non_empty_columns) == 0:
        raise ValueError("SMART_SLICE_ERROR: No character pixels found in image.")

    split_indices = np.where(np.diff(non_empty_columns) > 1)[0] + 1
    character_column_groups = np.split(non_empty_columns, split_indices)

    detected_count = len(character_column_groups)
    if detected_count == 6:
        sliced_frames = []
        for group in character_column_groups:
            start_x = int(group[0])
            end_x = int(group[-1]) + 1
            frame = transparent_image.crop((start_x, 0, end_x, transparent_image.height))
            sliced_frames.append(frame)
        return sliced_frames

    # 6개가 아니면 균등 6등분 폴백 (AI가 포즈를 붙여 그려서 덩어리가 3개 등으로 나온 경우)
    return _equal_slice_6_frames(transparent_image)


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

    # 스마트 슬라이싱: 알파 채널로 캐릭터 경계 감지 후 6프레임 분리 (6개 아니면 ValueError)
    try:
        original_frames = smart_slice_6_frames(img)
    except ValueError:
        raise

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
            raise ValueError("SMART_SLICE_ERROR: Empty frame after crop (character image is empty).")

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
