from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from src.utils.calibration import FHD_1080
from src.utils.image_utils import relative_roi_frame, to_gray
from src.utils.ocr_utils import normalize_ocr_text

HOME_GACHA_OCR_REFERENCE_ROI = (110, 993, 95, 54)
HOME_GACHA_OCR_RELATIVE_ROI = (
    110 / FHD_1080.width,
    993 / FHD_1080.height,
    205 / FHD_1080.width,
    1047 / FHD_1080.height,
)
HOME_GACHA_OCR_KEYWORD = "抽抽乐"
HOME_GACHA_OCR_SCALES = (1.0, 2.0, 3.0)
HOME_GACHA_OCR_ALIASES = (HOME_GACHA_OCR_KEYWORD,)
HOME_ANNOUNCEMENT_CLEAR_REFERENCE_POINT = (169, 615)
HOME_ANNOUNCEMENT_CLEAR_RELATIVE_POINT = (
    HOME_ANNOUNCEMENT_CLEAR_REFERENCE_POINT[0] / FHD_1080.width,
    HOME_ANNOUNCEMENT_CLEAR_REFERENCE_POINT[1] / FHD_1080.height,
)

# 左上图标列整列大 ROI（我的小屋/格鲁TALK/街机游戏）。检测模型需要上下文，
# 禁止单独拆出单标签小 ROI（实测丢字/返回空）。来源：BUG-20260829-01 实测标定。
HOME_LEFT_COLUMN_OCR_REFERENCE_ROI = (110, 165, 430, 155)
HOME_LEFT_COLUMN_OCR_RELATIVE_ROI = (
    110 / FHD_1080.width,
    165 / FHD_1080.height,
    540 / FHD_1080.width,
    320 / FHD_1080.height,
)
# 每组关键词为简体（OCR 只识别简体中文，2026-08-29 取消繁体识别别名）；
# 子串命中即该组计 1 票。
HOME_LEFT_COLUMN_KEYWORD_GROUPS = (
    ("我的小屋",),
    ("格鲁TALK",),
    ("街机游戏",),
)
HOME_LEFT_COLUMN_REQUIRED_HITS = 2

# 左列灰度 p95：未压暗实测定标 253、0.5x 公告压暗 126，阈值 185 干净分离，
# 且不随主页背景场景变化（旧模板相对亮度比值依赖采集场景，已废弃）。
HOME_DIMMED_P95_THRESHOLD_DEFAULT = 185.0

# 遮挡主页的模态弹窗（登录加成、签到、公告板等）：面板自身亮、四周被压暗。
MODAL_PANEL_BRIGHTNESS_THRESHOLD = 150
MODAL_PANEL_MIN_AREA_RATIO = 0.06
MODAL_PANEL_MAX_AREA_RATIO = 0.85
MODAL_PANEL_MIN_SIDE_RATIO = 0.18
MODAL_PANEL_CENTER_RANGE = (0.20, 0.80, 0.15, 0.85)
MODAL_PANEL_DETECT_MAX_SIDE = 960

# 关闭按钮不再按面板的固定比例估算。2026-11-16 实机对比两种弹窗：
#   「公告板」面板 1600x1220，X 中心在面板内 (0.960, 0.058)；
#   「登录加成」面板 1081x641，X 中心在面板内 (0.977, 0.160)。
# 两者纵向差 3 倍（头部高度不同），用同一个比例必然点到第一条公告上，
# 于是「打开公告 -> 退出 -> 再打开」的死循环。改成在面板右上角区域直接找
# 叉号字形：用局部中值背景差分，深色 ✕（白底）和浅色 ✕（彩色底）都命中。
MODAL_CLOSE_GLYPH_ROI = (0.60, 0.0, 1.0, 0.26)
MODAL_CLOSE_GLYPH_CONTRAST = 32
MODAL_CLOSE_GLYPH_MEDIAN_RATIO = 0.035
MODAL_CLOSE_GLYPH_SIDE_RATIO = (0.010, 0.050)
MODAL_CLOSE_GLYPH_ASPECT_RANGE = (0.6, 1.7)
MODAL_CLOSE_GLYPH_FILL_RANGE = (0.10, 0.65)
# 距面板右上角太近的连通域是圆角/描边伪影，不是按钮字形。
MODAL_CLOSE_GLYPH_EDGE_INSET = (0.018, 0.030)
# 仅在没有传入画面（旧调用/离线标定）时使用的估算比例。
MODAL_CLOSE_RELATIVE_IN_PANEL = (0.977, 0.167)

# 「新闻/公告」整页（从主页公告横幅/活动入口进入）：画面几乎铺满且亮度均匀，
# 不是居中小面板，modal_panel_bbox 会因面积超限拒绝，固定清理点又落在左侧
# 分类栏上（只会切换分类，退不出去）。唯一可靠的退出控件是左上角返回箭头。
# 2026-11-16 实机帧标定（2560x1440 抓帧，游戏内 3840x2160 等比）：
# 标题「新闻／公告」OCR 框 (461,58,420,80)，返回箭头深色笔画 bbox
# (328,73,29,54)、中心 (342,100) -> 相对 (0.1338, 0.0694)；
# 箭头 x 约等于「标题左边界 - 1.5 倍标题高度」，因此有 OCR 框时按标题反推，
# 没有 OCR 框时退回固定相对点（相对坐标与分辨率无关）。
NEWS_PAGE_TITLE_KEYWORD = "新闻公告"
NEWS_PAGE_SUPPORT_KEYWORDS = ("公告事项", "开发者笔记", "查看详情")
NEWS_BACK_REFERENCE_POINT = (342, 100)
NEWS_BACK_REFERENCE_SIZE = (2560, 1440)
NEWS_BACK_RELATIVE_POINT = (
    NEWS_BACK_REFERENCE_POINT[0] / NEWS_BACK_REFERENCE_SIZE[0],
    NEWS_BACK_REFERENCE_POINT[1] / NEWS_BACK_REFERENCE_SIZE[1],
)
NEWS_BACK_TITLE_OFFSET_RATIO = 1.5
NEWS_TITLE_MAX_RELATIVE_Y = 0.15


@dataclass(frozen=True)
class HomeGachaOcrResult:
    text: str
    selected_scale: float | None
    attempts: tuple[tuple[float, str], ...]

    @property
    def matched(self) -> bool:
        return self.selected_scale is not None

    @property
    def trace(self) -> str:
        attempts = ", ".join(
            f"x{scale:g}={text or '-'}" for scale, text in self.attempts
        )
        selected = (
            f"采用x{self.selected_scale:g}"
            if self.selected_scale is not None
            else "未命中"
        )
        return f"{selected}; {attempts}"


def home_gacha_ocr_with_fallback(
    read_text: Callable[[float], object],
    scales: tuple[float, ...] = HOME_GACHA_OCR_SCALES,
) -> HomeGachaOcrResult:
    """Read the fixed gacha ROI with bounded same-frame upscale fallbacks."""
    attempts: list[tuple[float, str]] = []
    first_nonempty = ""
    for scale in scales:
        text = str(read_text(float(scale)) or "").strip()
        attempts.append((float(scale), text))
        if text and not first_nonempty:
            first_nonempty = text
        if home_gacha_ocr_matches(text):
            return HomeGachaOcrResult(text, float(scale), tuple(attempts))
    return HomeGachaOcrResult(first_nonempty, None, tuple(attempts))


def home_gacha_ocr_matches(text: object) -> bool:
    normalized_text = normalize_ocr_text(text)
    return any(
        normalize_ocr_text(keyword) in normalized_text
        for keyword in HOME_GACHA_OCR_ALIASES
    )


def home_left_column_hits(text: object) -> int:
    """Count left-column keyword groups present in the same-frame OCR text."""
    normalized_text = normalize_ocr_text(text)
    hits = 0
    for aliases in HOME_LEFT_COLUMN_KEYWORD_GROUPS:
        if any(
            normalize_ocr_text(alias) in normalized_text
            for alias in aliases
        ):
            hits += 1
    return hits


def home_left_column_p95_brightness(frame) -> float:
    """Scene-independent 95th-percentile grayscale of the left-column ROI."""
    if frame is None:
        return 0.0
    _left, _top, crop = relative_roi_frame(frame, HOME_LEFT_COLUMN_OCR_RELATIVE_ROI)
    if crop.size == 0:
        return 0.0
    return float(np.percentile(to_gray(crop), 95))


def home_confirmation_passes(
    *,
    left_hits: int,
    required_left_hits: int,
    brightness: float,
    brightness_threshold: float,
    gacha_ocr_text: object,
) -> bool:
    """Require all three same-frame signals before confirming the global home page.

    brightness is a scene-independent dimming metric compared against
    brightness_threshold: the OCR pipeline passes the left-column grayscale
    p95 (0-255, threshold 185); the legacy template bridge passes the
    template-relative brightness ratio (0-1, threshold 0.75).
    """
    return (
        int(left_hits) >= int(required_left_hits)
        and float(brightness) >= float(brightness_threshold)
        and home_gacha_ocr_matches(gacha_ocr_text)
    )


def home_temporary_announcement_detected(
    *,
    left_hits: int,
    required_left_hits: int,
    brightness: float,
    brightness_threshold: float,
    gacha_ocr_text: object,
) -> bool:
    """Detect a dimmed global home page covered by a temporary announcement."""
    return (
        int(left_hits) >= int(required_left_hits)
        and float(brightness) < float(brightness_threshold)
        and home_gacha_ocr_matches(gacha_ocr_text)
    )


def modal_panel_bbox(frame) -> tuple[int, int, int, int] | None:
    """Return the (left, top, width, height) of a centred bright modal panel.

    A popup that covers the home screen renders as one large bright rectangle on
    an otherwise dimmed frame, so a brightness mask plus one morphological pass
    is enough.  The panel has to cross the frame centre and stay inside the
    configured area range; anything else (HUD numbers, side banners, artwork
    strips) is rejected so callers never click a guessed position.
    Detection runs on a downscaled copy because the mask work is O(pixels) and
    this path only fires while the home page is dimmed.
    """
    if frame is None:
        return None
    gray = to_gray(frame)
    if gray.size == 0:
        return None
    height, width = gray.shape[:2]
    if height <= 0 or width <= 0:
        return None

    scale = min(1.0, MODAL_PANEL_DETECT_MAX_SIDE / float(max(height, width)))
    work = gray
    if scale < 1.0:
        work = cv2.resize(
            gray,
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
    work_height, work_width = work.shape[:2]
    if work_height <= 0 or work_width <= 0:
        return None

    mask = (work >= MODAL_PANEL_BRIGHTNESS_THRESHOLD).astype(np.uint8)
    kernel_size = max(3, int(round(min(work_height, work_width) * 0.02)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    try:
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            mask, 8,
        )
    except cv2.error:
        return None

    frame_area = float(work_height * work_width)
    min_center_x, max_center_x, min_center_y, max_center_y = (
        MODAL_PANEL_CENTER_RANGE
    )
    best_area = 0.0
    best_bbox: tuple[int, int, int, int] | None = None
    for index in range(1, count):
        left, top, panel_width, panel_height, area = (
            int(value) for value in stats[index][:5]
        )
        if panel_width <= 0 or panel_height <= 0:
            continue
        area_ratio = float(area) / frame_area
        if not (
            MODAL_PANEL_MIN_AREA_RATIO <= area_ratio <= MODAL_PANEL_MAX_AREA_RATIO
        ):
            continue
        if (
            panel_width / work_width < MODAL_PANEL_MIN_SIDE_RATIO
            or panel_height / work_height < MODAL_PANEL_MIN_SIDE_RATIO
        ):
            continue
        center_x = (left + panel_width / 2) / work_width
        center_y = (top + panel_height / 2) / work_height
        if not (min_center_x <= center_x <= max_center_x):
            continue
        if not (min_center_y <= center_y <= max_center_y):
            continue
        # The panel must cover the frame centre, not sit off to one side.
        if not (
            left < work_width * 0.5 < left + panel_width
            and top < work_height * 0.5 < top + panel_height
        ):
            continue
        if float(area) > best_area:
            best_area = float(area)
            best_bbox = (left, top, panel_width, panel_height)
    if best_bbox is None or scale >= 1.0:
        return best_bbox

    left, top, panel_width, panel_height = best_bbox
    return (
        min(width - 1, int(round(left / scale))),
        min(height - 1, int(round(top / scale))),
        max(1, int(round(panel_width / scale))),
        max(1, int(round(panel_height / scale))),
    )


def modal_close_glyph(
    frame,
    bbox: tuple[int, int, int, int] | None,
) -> tuple[float, float] | None:
    """Return the relative position of the panel's ✕ close button.

    The button is a small, roughly square glyph near the panel's top-right
    corner.  Instead of assuming one fixed inset (different popups have
    different header heights and the guess lands on content), the glyph is
    found by local-contrast blobs: the crop is compared against its own median
    background, so a dark ✕ on white and a light ✕ on a coloured bar both
    survive.  Corner-radius shadows are dropped by an edge-inset rule.

    Returns ``None`` when no confident glyph exists — callers must then not
    click at all instead of guessing, because a wrong click lands on the
    popup's own list rows and opens them.
    """
    if frame is None or bbox is None:
        return None
    gray = to_gray(frame)
    if gray.size == 0:
        return None
    frame_height, frame_width = gray.shape[:2]
    try:
        left, top, panel_width, panel_height = (int(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if panel_width <= 0 or panel_height <= 0:
        return None

    roi_x0 = max(0, left + int(panel_width * MODAL_CLOSE_GLYPH_ROI[0]))
    roi_x1 = min(frame_width, left + int(panel_width * MODAL_CLOSE_GLYPH_ROI[2]))
    roi_y0 = max(0, top + int(panel_height * MODAL_CLOSE_GLYPH_ROI[1]))
    roi_y1 = min(frame_height, top + int(panel_height * MODAL_CLOSE_GLYPH_ROI[3]))
    if roi_x1 - roi_x0 < 16 or roi_y1 - roi_y0 < 16:
        return None

    crop = gray[roi_y0:roi_y1, roi_x0:roi_x1]
    kernel = max(
        15, min(81, int(round(panel_width * MODAL_CLOSE_GLYPH_MEDIAN_RATIO)) | 1),
    )
    if min(crop.shape[:2]) <= kernel:
        return None
    background = cv2.medianBlur(crop, kernel).astype(np.int16)
    difference = np.abs(crop.astype(np.int16) - background)
    mask = (difference >= MODAL_CLOSE_GLYPH_CONTRAST).astype(np.uint8)
    try:
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            mask, 8,
        )
    except cv2.error:
        return None

    crop_height, crop_width = crop.shape[:2]
    min_side, max_side = MODAL_CLOSE_GLYPH_SIDE_RATIO
    min_aspect, max_aspect = MODAL_CLOSE_GLYPH_ASPECT_RANGE
    min_fill, max_fill = MODAL_CLOSE_GLYPH_FILL_RANGE
    inset_x, inset_y = MODAL_CLOSE_GLYPH_EDGE_INSET
    best_key: float | None = None
    best_point: tuple[float, float] | None = None
    for index in range(1, count):
        x, y, width, height, area = (
            int(value) for value in stats[index][:5]
        )
        if width <= 0 or height <= 0:
            continue
        # 贴住裁剪边界的是面板边框/圆角，不是按钮。
        if (
            x <= 1
            or y <= 1
            or x + width >= crop_width - 1
            or y + height >= crop_height - 1
        ):
            continue
        side = max(width, height)
        if not (min_side * panel_width <= side <= max_side * panel_width):
            continue
        aspect = width / float(height)
        if not (min_aspect <= aspect <= max_aspect):
            continue
        fill = area / float(width * height)
        if not (min_fill <= fill <= max_fill):
            continue
        center_x = roi_x0 + x + width / 2.0
        center_y = roi_y0 + y + height / 2.0
        relative_x = (center_x - left) / float(panel_width)
        relative_y = (center_y - top) / float(panel_height)
        if (1.0 - relative_x) < inset_x or relative_y < inset_y:
            continue
        # 关闭按钮是离右上角最近的那个字形。
        key = (left + panel_width - center_x) + (center_y - top) * 0.5
        if best_key is None or key < best_key:
            best_key = key
            best_point = (
                min(1.0, max(0.0, center_x / frame_width)),
                min(1.0, max(0.0, center_y / frame_height)),
            )
    return best_point


def modal_close_point(
    bbox: tuple[int, int, int, int] | None,
    width: int,
    height: int,
    *,
    frame=None,
) -> tuple[float, float] | None:
    """Map a modal panel bbox to the relative position of its close button.

    With ``frame`` the position comes from the ✕ glyph itself and returns
    ``None`` when it cannot be located (never guess).  Without ``frame`` the
    legacy fixed inset is used, which stays for offline calibration only.
    """
    if bbox is None or width <= 0 or height <= 0:
        return None
    left, top, panel_width, panel_height = (float(value) for value in bbox)
    if panel_width <= 0 or panel_height <= 0:
        return None
    if frame is not None:
        return modal_close_glyph(frame, bbox)
    offset_x, offset_y = MODAL_CLOSE_RELATIVE_IN_PANEL
    x = left + panel_width * offset_x
    y = top + panel_height * offset_y
    return (
        min(1.0, max(0.0, x / width)),
        min(1.0, max(0.0, y / height)),
    )


def news_page_detected(ocr_text: object) -> bool:
    """Detect the full-screen 新闻/公告 page (a sub page, not a modal).

    Only OCR evidence is used: the merged title reads 新闻／公告 and the left
    category column always carries 公告事项 / 开发者笔记, so the check cannot
    fire on the home screen where none of those strings exist.
    """
    text = normalize_ocr_text(ocr_text, alnum_only=True)
    if not text:
        return False
    if NEWS_PAGE_TITLE_KEYWORD in text:
        return True
    if "新闻" not in text or "公告" not in text:
        return False
    return any(keyword in text for keyword in NEWS_PAGE_SUPPORT_KEYWORDS)


def _box_geometry(box) -> tuple[float, float, float, float] | None:
    values = [getattr(box, key, None) for key in ("x", "y", "width", "height")]
    if any(value is None for value in values):
        raw = getattr(box, "box", None)
        if raw is None or len(raw) < 4:
            return None
        values = list(raw[:4])
    try:
        x, y, width, height = (float(value) for value in values)
    except (TypeError, ValueError):
        return None
    return x, y, width, height


def news_back_point(
    boxes: object = None,
    width: int = 0,
    height: int = 0,
) -> tuple[float, float]:
    """Return the relative position of the 新闻/公告 back arrow.

    The arrow sits on the same row as the title, just left of it, so the OCR
    title box gives a scale-independent anchor. Without a usable title box the
    calibrated constant is used instead of guessing a fresh position.
    """
    if boxes and width > 0 and height > 0:
        best: tuple[float, float] | None = None
        for box in boxes:
            name = normalize_ocr_text(
                getattr(box, "name", ""), alnum_only=True,
            )
            if "新闻" not in name and "公告" not in name:
                continue
            geometry = _box_geometry(box)
            if geometry is None:
                continue
            x, y, _box_width, box_height = geometry
            if box_height <= 0:
                continue
            center_y = (y + box_height / 2) / height
            if center_y > NEWS_TITLE_MAX_RELATIVE_Y:
                continue
            candidate_x = (x - NEWS_BACK_TITLE_OFFSET_RATIO * box_height) / width
            if best is None or center_y < best[1]:
                best = (candidate_x, center_y)
        if best is not None and 0.0 < best[0] < 0.5:
            return best
    return NEWS_BACK_RELATIVE_POINT
