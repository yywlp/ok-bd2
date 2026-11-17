"""主线「快速探索」自动刷取流程。

实机流程（1920x1080，界面用词取自实机截图）：

    关卡内：右上角任务面板下方有「进行快速探索」
        ↓ 点击
    关卡任务列表：标题形如「血骑士-极难」，右下角一个长条按钮
        ├ 按钮为「自动进行」→ 点击，游戏自动推进非战斗节点
        └ 按钮为「战斗」    → 点击，进入战斗
        ↓ 战斗结束
    结算页：「MISSION COMPLETED」+ 右下角「离开」
        ↓ 点击
    回到关卡内，继续下一轮

节点顺序不固定，中间还夹杂剧情、掉落等界面，因此每一步都以**当前帧的按钮文字**
为准，不做固定脚本；认不出来就等，等不到就停下来存证，绝不盲点。
"""

from __future__ import annotations

from enum import Enum
from time import monotonic

from ok.task.exceptions import FinishedException, TaskDisabledException

from src.tasks.task_vision_mixin import REFERENCE_HEIGHT, REFERENCE_WIDTH
from src.utils.image_utils import crop_relative
from src.utils.ocr_utils import keyword_match_count, normalize_ocr_text


def _main_story_roi(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> tuple[float, float, float, float]:
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    return (
        left / REFERENCE_WIDTH,
        top / REFERENCE_HEIGHT,
        right / REFERENCE_WIDTH,
        bottom / REFERENCE_HEIGHT,
    )


# 关卡内右上角「进行快速探索」按钮。实测文字框 (1575,192)-(1861,228)。

IN_STAGE_BUTTON_ROI = _main_story_roi(1555, 180, 1885, 242)

IN_STAGE_BUTTON_POINT = (1718, 210)

# 关卡任务列表右下角长条按钮。实测「自动进行」条 (1488,984)-(1791,1041)，
# 「战斗」条 (1493,980)-(1792,1037)，文字都落在中间这一段。
LIST_BUTTON_ROI = _main_story_roi(1580, 972, 1825, 1048)

LIST_BUTTON_POINT = (1655, 1010)

# 列表页左上角标题，形如「血骑士-极难」，用于关卡校验与难度核对。
# 左边留出 210 是为了避开返回箭头按钮，只框盾牌图标与标题文字。
LIST_TITLE_ROI = _main_story_roi(210, 12, 470, 60)

# 战斗结算页右下角「离开」按钮。实测 (1609,983)-(1825,1028)。
VICTORY_LEAVE_ROI = _main_story_roi(1588, 963, 1846, 1042)

VICTORY_LEAVE_POINT = (1717, 1004)

# 奖励弹窗：中部「可获得奖励」标题 + 底部「点击画面即可返回」提示。
# 实测文字框 (1192,789)-(1306,816) 与 (866,975)-(1040,1004)，取中间并稍放宽。
REWARD_POPUP_TITLE_ROI = _main_story_roi(1150, 760, 1350, 850)
REWARD_POPUP_HINT_ROI = _main_story_roi(800, 940, 1100, 1030)

# 弹窗关闭点：底部「点击画面即可返回」文字所在区域，点它即关闭弹窗。
REWARD_POPUP_CLOSE_POINT = (953, 989)

# 结算页顶部标题，用于区分「真结算」与其它带「离开」的弹窗。
VICTORY_TITLE_ROI = _main_story_roi(1250, 20, 1900, 210)

IN_STAGE_KEYWORDS = ("进行快速探索",)

AUTO_KEYWORDS = ("自动进行",)

BATTLE_KEYWORDS = ("战斗",)

VICTORY_KEYWORDS = ("离开",)

# 战斗内 HUD：右下角与结算按钮是同一块屏幕区域，2026-11-16 实机日志里
# 「结束自动战斗esc」被「战斗」关键字命中、随后「TURN.1 / BATTLE」被当成
# 未知列表按钮直接停机。战斗内必须先于按钮判定被识别出来。
# 简体客户端里 BATTLE 只在战斗内 HUD 出现，列表页那个按钮是中文「战斗」。
BATTLE_HUD_KEYWORDS = ("结束自动战斗", "battle")

# 奖励弹窗：战斗结算或节点完成后的「可获得奖励」+「点击画面即可返回」。
# 2026-11-17 实机日志：列表页中部常驻「可获得奖励」奖励预览标签，它的位置
# （相对 0.620~0.680 / 0.730~0.754）和弹窗标题的标定 ROI 完全重叠。只按
# 「可获得奖励」判定会把整张列表页当成弹窗，于是点底部空地直到超时。
# 弹窗独有的证据只有底部提示文字，因此只认「点击画面即可返回」。
REWARD_POPUP_KEYWORDS = ("点击画面即可返回",)

# 提示文字较长，OCR 偶尔错一字，允许一定模糊度。
REWARD_POPUP_FUZZY_RATIO = 0.7

# 「列表按钮未知」要持续这么久才真停机，避免单帧 OCR 噪声直接终止整轮。
LIST_OTHER_CONFIRM_SECONDS = 3.0


def _looks_like_battle_hud(*texts: object) -> bool:
    """判断这几块 ROI 的文字像不像战斗内 HUD（而非列表页按钮）。"""

    merged = "".join(normalize_ocr_text(text) for text in texts)
    if not merged:
        return False
    return any(keyword in merged for keyword in BATTLE_HUD_KEYWORDS)


def _looks_like_reward_popup(*texts: object) -> bool:
    """判断这几块 ROI 的文字像不像奖励弹窗（只认弹窗独有的底部提示）。"""

    merged = "".join(normalize_ocr_text(text) for text in texts)
    if not merged:
        return False
    return keyword_match_count(
        merged,
        REWARD_POPUP_KEYWORDS,
        fuzzy_ratio=REWARD_POPUP_FUZZY_RATIO,
    ) >= 1


# OCR 文本短、字形相近，允许一定模糊度；0.75 对 4 字词只容忍 1 字差异。
BUTTON_FUZZY_RATIO = 0.75


class MainStoryState(str, Enum):
    IN_STAGE = "关卡内"
    LIST_AUTO = "列表-自动进行"
    LIST_BATTLE = "列表-战斗"
    LIST_OTHER_BUTTON = "列表-按钮未知"
    VICTORY = "战斗结算"
    IN_BATTLE = "战斗内"
    REWARD_POPUP = "奖励弹窗"
    BUSY = "其他界面"


def classify_main_story_state(
    in_stage_text: object,
    button_text: object,
    victory_text: object,
    reward_text: object,
) -> MainStoryState:
    """把几块 ROI 的 OCR 文本归到一个界面状态。

    结算页优先判定：战斗刚结束时列表页已经消失，「离开」只出现在结算页右下角，
    因此这一块 ROI 读到「离开」即可认定结算，不再要求标题同时命中——
    结算页的标题是艺术字，OCR 命中率远不如按钮文字稳定。

    奖励弹窗优先于按钮判定：弹窗出现时「自动进行」仍在列表页 ROI 内可见，
    若把「自动进行」当列表页处理，点击只会把弹窗点掉，不会真正推进。
    """

    if keyword_match_count(victory_text, VICTORY_KEYWORDS):
        return MainStoryState.VICTORY

    if keyword_match_count(in_stage_text, IN_STAGE_KEYWORDS, fuzzy_ratio=BUTTON_FUZZY_RATIO):
        return MainStoryState.IN_STAGE

    # 战斗内的右下角 HUD 与列表按钮共用同一块区域，必须在按钮词表之前拦下，
    # 否则「结束自动战斗esc」会被当成「战斗」按钮、「TURN.1 BATTLE」会被
    # 当成未知按钮而触发停机。
    if _looks_like_battle_hud(button_text, victory_text):
        return MainStoryState.IN_BATTLE

    # 奖励弹窗可能盖住列表页，必须在按钮词表之前拦下。
    if _looks_like_reward_popup(reward_text):
        return MainStoryState.REWARD_POPUP

    if keyword_match_count(button_text, AUTO_KEYWORDS, fuzzy_ratio=BUTTON_FUZZY_RATIO):
        return MainStoryState.LIST_AUTO

    if keyword_match_count(button_text, BATTLE_KEYWORDS, fuzzy_ratio=BUTTON_FUZZY_RATIO):
        return MainStoryState.LIST_BATTLE

    if normalize_ocr_text(button_text):
        return MainStoryState.LIST_OTHER_BUTTON

    return MainStoryState.BUSY


class MainStoryConfigMixin:
    """主线刷取的配置项，键名统一以「主线」开头，避免与日常任务冲突。"""

    MAIN_STORY_CONFIG_KEYS = (
        "主线关卡标题校验",
        "主线战斗次数上限",
        "主线时间上限分钟",
        "主线等待秒数",
        "主线战斗等待秒数",
        "主线 OCR 阈值",
        "主线未知界面时停止",
    )

    def _install_main_story_config(self) -> None:
        self.default_config.update(
            {
                "启用": True,
                "主线关卡标题校验": "",
                "主线战斗次数上限": 0,
                "主线时间上限分钟": 0.0,
                "主线等待秒数": 20.0,
                "主线战斗等待秒数": 120.0,
                "主线 OCR 阈值": 0.2,
                "主线未知界面时停止": True,
            }
        )
        self.config_description.update(
            {
                "启用": "是否执行主线快速探索刷取。",
                "主线关卡标题校验": (
                    "关卡标题应包含的文字（如「血骑士-极难」）。留空则不校验；"
                    "填了但列表页标题对不上时会拒绝开始，避免刷错关卡。"
                ),
                "主线战斗次数上限": "完成多少次「战斗」后停止，0 表示不限。",
                "主线时间上限分钟": "运行多久后停止，0 表示不限。",
                "主线等待秒数": "每次点击后等待界面变化的最长秒数。",
                "主线战斗等待秒数": (
                    "进入战斗后等待结算的最长秒数；期间认不出界面属于正常，超时才会停止。"
                ),
                "主线 OCR 阈值": "按钮与标题文字识别的最低置信度。",
                "主线未知界面时停止": (
                    "开启时，遇到既不是关卡内、也不是列表或结算的界面会停机并存图；"
                    "关闭则一直等待到超时。"
                ),
            }
        )
        self.config_type.update(
            {
                "主线战斗次数上限": {"min": 0, "max": 999, "step": 1},
                "主线时间上限分钟": {"min": 0.0, "max": 1440.0, "step": 5.0},
                "主线等待秒数": {"min": 5.0, "max": 120.0, "step": 1.0},
                "主线战斗等待秒数": {"min": 20.0, "max": 600.0, "step": 10.0},
                "主线 OCR 阈值": {"min": 0.05, "max": 0.95, "step": 0.01},
            }
        )


class MainStoryFeatureMixin:
    """主线刷取的状态机主体。"""

    def run_main_story_farm(self) -> bool:
        battle_limit = int(self.config.get("主线战斗次数上限", 0) or 0)
        minute_limit = float(self.config.get("主线时间上限分钟", 0.0) or 0.0)
        wait_seconds = float(self.config.get("主线等待秒数", 20.0))
        battle_wait = float(self.config.get("主线战斗等待秒数", 120.0))
        threshold = float(self.config.get("主线 OCR 阈值", 0.2))
        stop_on_unknown = bool(self.config.get("主线未知界面时停止", True))

        deadline = monotonic() + minute_limit * 60 if minute_limit > 0 else None

        battles = 0
        autos = 0
        cycles = 0
        busy_since: float | None = None
        busy_announced = False
        unknown_since: float | None = None

        self._main_story_set_stats(battles, autos, cycles)

        while True:
            if deadline is not None and monotonic() >= deadline:
                self.log_info("主线刷取：到达时间上限，结束。")
                self._main_story_set_stats(battles, autos, cycles)
                return True

            frame = self.capture_frame()
            state, evidence = self._main_story_state(frame, threshold)
            self.info_set("主线当前界面", f"{state.value}｜{evidence}")

            if state is MainStoryState.IN_STAGE:
                busy_since, busy_announced = None, False
                unknown_since = None
                # 关卡内看不到列表标题（那块区域是地图），此处不做关卡校验，
                # 校验统一放到列表页做。
                self.log_info("主线刷取：打开快速探索列表。")
                self._click_reference(*IN_STAGE_BUTTON_POINT)
                if self._main_story_wait_change(state, wait_seconds, threshold):
                    continue
                return self._main_story_stuck(frame, "点击「进行快速探索」后界面无变化")

            if state is MainStoryState.LIST_AUTO:
                busy_since, busy_announced = None, False
                unknown_since = None
                if not self._main_story_verify_title(frame, threshold):
                    return self._main_story_abort(frame, "关卡标题校验未通过")
                self.log_info("主线刷取：自动进行。")
                self._click_reference(*LIST_BUTTON_POINT)
                autos += 1
                self._main_story_set_stats(battles, autos, cycles)
                # 自动进行可能要跑完一串节点才回到列表或进战斗，用战斗等待兜底。
                if self._main_story_wait_change(state, battle_wait, threshold):
                    continue
                return self._main_story_stuck(frame, "点击「自动进行」后界面长时间无变化")

            if state is MainStoryState.LIST_BATTLE:
                busy_since, busy_announced = None, False
                unknown_since = None
                if battle_limit and battles >= battle_limit:
                    self.log_info(f"主线刷取：已达战斗次数上限 {battle_limit}，结束。")
                    self._main_story_set_stats(battles, autos, cycles)
                    return True
                if not self._main_story_verify_title(frame, threshold):
                    return self._main_story_abort(frame, "关卡标题校验未通过")
                self.log_info(f"主线刷取：第 {battles + 1} 场战斗。")
                self._click_reference(*LIST_BUTTON_POINT)
                battles += 1
                self._main_story_set_stats(battles, autos, cycles)
                if self._main_story_wait_change(state, battle_wait, threshold):
                    continue
                return self._main_story_stuck(frame, "点击「战斗」后未等到结算")

            if state is MainStoryState.VICTORY:
                busy_since, busy_announced = None, False
                unknown_since = None
                self.log_info("主线刷取：结算完成，离开战斗。")
                self._click_reference(*VICTORY_LEAVE_POINT)
                cycles += 1
                self._main_story_set_stats(battles, autos, cycles)
                if self._main_story_wait_change(state, wait_seconds, threshold):
                    continue
                return self._main_story_stuck(frame, "点击「离开」后界面无变化")

            if state is MainStoryState.REWARD_POPUP:
                busy_since, busy_announced = None, False
                unknown_since = None
                self.log_info("主线刷取：奖励弹窗，点击关闭。")
                self._click_reference(*REWARD_POPUP_CLOSE_POINT)
                if self._main_story_wait_change(state, wait_seconds, threshold):
                    continue
                return self._main_story_stuck(frame, "点击「奖励弹窗」后界面无变化")

            if state is MainStoryState.LIST_OTHER_BUTTON:
                # 列表页还在，但按钮既不是自动进行也不是战斗：通常意味着本关已无
                # 可推进的节点，或是游戏新增了未知按钮。停下来比乱点安全。
                # 但单帧 OCR 噪声也会落进这里，所以要求持续确认一段时间才真停。
                if self.config.get("主线未知界面时停止", True):
                    if unknown_since is None:
                        unknown_since = monotonic()
                        busy_since, busy_announced = None, False
                        self.sleep(0.5)
                        continue
                    if monotonic() - unknown_since >= LIST_OTHER_CONFIRM_SECONDS:
                        return self._main_story_abort(
                            frame, "列表按钮文字未知（可能已无可刷节点）",
                        )
                    self.sleep(0.5)
                    continue
                unknown_since = None
                busy_since, busy_announced = None, False
                self.sleep(1.0)
                continue

            # BUSY / IN_BATTLE：战斗中、剧情中、掉落弹窗等。战斗期间认不出界面
            # 是正常的，用「主线战斗等待秒数」兜底，超时才认为真卡住了。
            # IN_BATTLE 特意不走独立分支，就是为了让它沿用同一套超时兜底。
            unknown_since = None
            now = monotonic()
            if busy_since is None:
                busy_since = now
                if not busy_announced:
                    busy_announced = True
                    self.log_info("主线刷取：等待界面推进。")
            elif now - busy_since > battle_wait:
                if stop_on_unknown:
                    return self._main_story_stuck(
                        frame, f"界面无推进超过 {battle_wait:.0f} 秒",
                    )
                busy_since, busy_announced = now, False

            self.sleep(0.5)

    def _main_story_state(
        self,
        frame,
        threshold: float,
    ) -> tuple[MainStoryState, str]:
        in_stage = self._main_story_ocr(frame, IN_STAGE_BUTTON_ROI, "主线快速探索入口", threshold)
        button = self._main_story_ocr(frame, LIST_BUTTON_ROI, "主线列表按钮", threshold)
        victory = self._main_story_ocr(frame, VICTORY_LEAVE_ROI, "主线结算离开", threshold)
        reward_title = self._main_story_ocr(frame, REWARD_POPUP_TITLE_ROI, "主线奖励标题", threshold)
        reward_hint = self._main_story_ocr(frame, REWARD_POPUP_HINT_ROI, "主线奖励提示", threshold)
        reward = normalize_ocr_text(reward_title + " " + reward_hint)
        # 标题只作存证，且仅在疑似结算时识别，省掉常态每帧一次 OCR。
        title = (
            self._main_story_ocr(frame, VICTORY_TITLE_ROI, "主线结算标题", threshold)
            if victory
            else ""
        )
        state = classify_main_story_state(in_stage, button, victory, reward)
        evidence = (
            f"入口={in_stage or '-'};按钮={button or '-'};"
            f"离开={victory or '-'};标题={title or '-'};奖励={reward or '-'}"
        )
        return state, evidence

    def _main_story_ocr(
        self,
        frame,
        bounds: tuple[float, float, float, float],
        name: str,
        threshold: float,
    ) -> str:
        crop = crop_relative(frame, bounds)
        if crop is None or crop.size == 0:
            return ""
        try:
            boxes = self.ocr(frame=crop, threshold=threshold, log=False, name=name)
        except (TaskDisabledException, FinishedException):
            raise
        except Exception as exc:
            self.info_set(f"{name} OCR错误", str(exc))
            return ""
        return normalize_ocr_text(" ".join(box.name for box in boxes if getattr(box, "name", "")))

    def _main_story_title_text(self, frame, threshold: float) -> str:
        return self._main_story_ocr(frame, LIST_TITLE_ROI, "主线关卡标题", threshold)

    def _main_story_verify_title(self, frame, threshold: float) -> bool:
        """只在列表页调用：关卡标题出现在列表页左上角，空白视为不认识。"""

        expected = normalize_ocr_text(self.config.get("主线关卡标题校验", ""))
        if not expected:
            return True
        actual = self._main_story_title_text(frame, threshold)
        # 允许一字之差：「士/土」「极/極」这类形近误识别不该导致整轮中止。
        if keyword_match_count(actual, [expected], fuzzy_ratio=0.8):
            return True
        self.info_set("主线关卡标题", actual or "<未识别>")
        return False

    def _main_story_wait_change(
        self,
        previous: MainStoryState,
        timeout: float,
        threshold: float,
    ) -> bool:
        """等界面离开 previous；超时返回 False。"""

        deadline = monotonic() + max(1.0, timeout)
        while monotonic() < deadline:
            self.sleep(0.5)
            try:
                frame = self.capture_frame()
            except Exception as exc:
                self.info_set("主线截图错误", str(exc))
                continue
            state, evidence = self._main_story_state(frame, threshold)
            if state is not previous:
                self.info_set("主线当前界面", f"{state.value}｜{evidence}")
                return True
        return False

    def _main_story_set_stats(self, battles: int, autos: int, cycles: int) -> None:
        self.info_set("主线战斗次数", battles)
        self.info_set("主线自动进行次数", autos)
        self.info_set("主线完成轮次", cycles)

    def _main_story_abort(self, frame, reason: str) -> bool:
        """安全的主动停止：存图存证，不点击。"""

        self.save_frame("main_story_abort", frame)
        self.info_set("主线停止原因", reason)
        self.log_info(f"主线刷取停止：{reason}")
        return False

    def _main_story_stuck(self, frame, reason: str) -> bool:
        self.save_frame("main_story_stuck", frame)
        self.info_set("主线停止原因", reason)
        self.log_error(f"主线刷取卡住：{reason}；已保存 probe_outputs/main_story_stuck.png")
        return False
