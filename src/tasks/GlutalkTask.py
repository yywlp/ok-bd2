"""格鲁TALK：检查可对话/未读会话，自动点开并推进聊天剧情。

真实界面（2026-11-15 实测）：GLU TALK 是一个聊天窗口，蓝色标题栏右侧显示
「今日的剩余相遇 N」，左上有「未读消息 N」，左栏是会话列表（未读行右端有蓝色
数字角标），右栏是消息与右下角的回复按钮。进入会话后点回复按钮即可推进剧情，
全屏剧情则靠「跳过 / SKIP」推进。

设计要点：
* 默认自动执行；「仅探测模式」可随时打开，只抓屏 + 记录 OCR 证据，不点击。
* 所有点击都要当帧的正面证据（未读汇总 + 角标 / 回复按钮 / 跳过文字或模板），
  证据不足一律安全停止，绝不靠坐标盲点。
* 运行时若已经在 GLU TALK 页面，直接开始检查，不再回主页重复点入口。
* 结束（成功、无任务、失败）都会尝试回主页，避免卡在子界面拖累后续任务。
"""

from __future__ import annotations

from datetime import datetime
from time import monotonic

from src.icons import TaskIcon
from src.tasks.BaseBD2Task import BaseBD2Task
from src.tasks.glutalk_vision import (
    AVAILABLE_TASK_DEFAULT_KEYWORDS,
    TalkAction,
    available_task_point,
    classify_talk_page,
    close_button_point,
    dialogue_decision,
    dialogue_ended,
    episode_number,
    home_talk_entry,
    remaining_encounters,
    reply_button_point,
    unread_badge_point,
    unread_count,
)
from src.tasks.task_vision_mixin import TaskVisionMixin
from src.utils.vision_models import TemplateSpec

# UI_Skip.png 的 ROI 沿用「首次卡带跳过」的实测值；UI_talk.png 为双箭头快进，
# 只在画面下半部搜索，避免把其他 UI 装饰误判为推进按钮。
SKIP_TEMPLATE = TemplateSpec(
    "格鲁TALK跳过", "image/UI_Skip.png", 0.72, roi=(1372, 14, 398, 102),
)
TALK_TEMPLATE = TemplateSpec(
    "格鲁TALK快进", "image/UI_talk.png", 0.72,
    relative_roi=(0.45, 0.68, 1.0, 1.0),
)


class GlutalkTask(TaskVisionMixin, BaseBD2Task):
    status_keys = [
        "启用", "状态", "当前阶段", "格鲁TALK OCR", "页面判断", "页面判定诊断",
        "剩余相遇", "未读消息", "可对话证据", "回复证据", "对话阶段", "完成对话数",
        "推进点击数", "探测结果", "Log", "Warning", "Error",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "格鲁TALK 对话"
        self.description = (
            "检查格鲁TALK 的未读/可对话会话并自动推进聊天剧情；"
            "需要采集界面证据时可临时打开「仅探测模式」。"
        )
        self.icon = TaskIcon.BOOK_SHELF
        self.group_name = "日常/周常"
        self.group_icon = TaskIcon.CALENDAR
        self.visible = True
        self._init_vision_state()
        self.default_config.update({
            "启用": True,
            "仅探测模式": False,
            "保存探测截图": True,
            "格鲁TALK OCR 阈值": 0.35,
            "主页确认等待秒数": 10.0,
            "页面等待秒数": 12.0,
            "扫描最长秒数": 180.0,
            "每段对话最长秒数": 90.0,
            "回复点击间隔秒数": 0.8,
            "最多对话次数": 3,
            "最多推进点击次数": 40,
            "每段最多点击次数": 12,
            "每段结束后回主页重进": True,
            "重进后等待秒数": 2.0,
            "无动作判定轮数": 6,
            "可对话关键词": "、".join(AVAILABLE_TASK_DEFAULT_KEYWORDS),
            "使用跳过按钮": True,
            "自动发送回复": True,
            "点击对话选项": False,
            "允许无证据点击推进": False,
            "回主页返回键最多次数": 3,
        })
        self.config_description.update({
            "仅探测模式": (
                "开启时只抓屏并记录 OCR 文本到 probe_outputs/glutalk_probe_latest.txt，"
                "不点击游戏。默认关闭，即自动检查并推进对话。"
            ),
            "保存探测截图": "仅探测模式下同时保存一张当前游戏画面。",
            "最多对话次数": "单次运行最多点开几个会话；官方每天最多 3 次会面活动。",
            "最多推进点击次数": "整个扫描流程的点击上限，防止界面卡住时无限点击。",
            "每段最多点击次数": "单个 EP 剧情内的点击上限，达到后安全停止该段。",
            "每段结束后回主页重进": (
                "出现「EPn. 对话结束」后，先返回主界面、再重新打开格鲁TALK，"
                "才会触发下一阶段对话（游戏机制）。关闭后只处理当前一段。"
            ),
            "重进后等待秒数": "重新打开格鲁TALK 后等待界面刷新的时间。",
            "无动作判定轮数": "连续多少帧找不到可点击元素就判定没有更多对话。",
            "可对话关键词": (
                "逗号/顿号分隔。只有条目文字命中其中之一且未命中「已完成」等否定词才会点击。"
                "留空则安全停止、不点击任何条目。"
            ),
            "使用跳过按钮": "识别到 UI_Skip / UI_talk 或「跳过」文字时点击以快进剧情。",
            "自动发送回复": (
                "格鲁TALK 聊天页右下角出现回复按钮时自动点击以推进剧情。"
                "关闭后只会点开未读会话，不会替你发送回复。"
            ),
            "点击对话选项": "对话出现选择/确认按钮时点击（默认关闭，避免误选剧情分支）。",
            "允许无证据点击推进": (
                "关闭时只按正面证据点击；开启后才允许无证据时点画面下半部推进。"
                "默认保持关闭。"
            ),
            "回主页返回键最多次数": "自动返回主页失败时按返回键逐级退出的最多次数。",
        })
        self._last_diagnostic = ""

    # ------------------------------------------------------------------ utils
    def _threshold(self) -> float:
        return max(0.05, min(0.95, float(self.config.get("格鲁TALK OCR 阈值", 0.35))))

    def _read(self):
        frame = self.capture_frame()
        boxes = list(self.ocr(
            frame=frame,
            threshold=self._threshold(),
            target_height=1080,
            log=False,
            name="格鲁TALK",
        ))
        text = " ".join(
            str(box.name) for box in boxes if getattr(box, "name", "")
        )
        self.info_set("格鲁TALK OCR", text[:600] or "-")
        return frame, boxes, text

    def _diagnose(self, reason: str) -> None:
        if reason and reason != self._last_diagnostic:
            self.info_set("页面判定诊断", reason)
            self._last_diagnostic = reason

    def _keywords(self) -> tuple[str, ...]:
        raw = str(self.config.get("可对话关键词", ""))
        return tuple(
            part.strip() for part in raw.replace("、", ",").replace(";", ",").split(",")
            if part.strip()
        )

    def _report_counters(self, text: str) -> tuple[int | None, int | None]:
        remaining = remaining_encounters(text)
        unread = unread_count(text)
        self.info_set("剩余相遇", "-" if remaining is None else str(remaining))
        self.info_set("未读消息", "-" if unread is None else str(unread))
        return remaining, unread

    def _safe_return_home(self) -> bool:
        self.info_set("当前阶段", "返回主页")
        if self.auto_return_main_home():
            return True
        return self.try_return_home_with_back(
            max_presses=int(self.config.get("回主页返回键最多次数", 3)),
        )

    def _fail(self, stage: str) -> bool:
        self.info_set("状态", f"格鲁TALK 失败：{stage}。")
        self.log_warning(f"格鲁TALK：{stage}，尝试返回主页。")
        self._safe_return_home()
        return False

    # ------------------------------------------------------------------ probe
    def _probe(self) -> bool:
        self.info_set("状态", "仅探测模式：只采集界面证据，不点击。")
        frame, boxes, text = self._read()
        height, width = frame.shape[:2]
        view = classify_talk_page(text)
        self.info_set("页面判断", view.page)
        self._diagnose(view.diagnostics)
        remaining, unread = self._report_counters(text)
        reply, reply_evidence = reply_button_point(boxes, width, height)
        badge, badge_evidence = unread_badge_point(
            boxes, width, height, unread=unread,
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if bool(self.config.get("保存探测截图", True)):
            self.save_frame(f"glutalk_probe_{timestamp}", frame)
        lines = [
            f"timestamp={timestamp}",
            f"capture_method={self.capture_method_name}",
            f"resolution={width}x{height}",
            f"text_count={len(boxes)}",
            f"page={view.page}",
            f"diagnostics={view.diagnostics}",
            f"remaining_encounters={remaining}",
            f"unread_messages={unread}",
            f"reply_button={reply} ({reply_evidence})",
            f"unread_badge={badge} ({badge_evidence})",
            f"available_keywords={self._keywords()}",
            "",
            "== OCR TEXT ==",
            text or "-",
            "",
            "== BOXES (name / confidence / x,y,w,h) ==",
        ]
        for box in boxes:
            lines.append(
                f"{getattr(box, 'name', '')}\t{getattr(box, 'confidence', 0):.3f}\t"
                f"{getattr(box, 'x', 0)},{getattr(box, 'y', 0)},"
                f"{getattr(box, 'width', 0)},{getattr(box, 'height', 0)}"
            )
        output = self.write_probe_text("glutalk_probe_latest.txt", lines)
        self.info_set("探测结果", str(output))
        self.log_completion(
            f"格鲁TALK 仅探测完成：页面={view.page}，剩余相遇={remaining}，"
            f"未读消息={unread}，证据文件={output}。关闭「仅探测模式」即可自动执行。"
        )
        return True

    # ------------------------------------------------------------------ flow
    def _wait_for_page(self, page: str, timeout: float) -> bool:
        end_at = monotonic() + max(1.0, timeout)
        while monotonic() < end_at:
            _, _, text = self._read()
            view = classify_talk_page(text)
            self.info_set("页面判断", view.page)
            self._diagnose(view.diagnostics)
            if view.page == page:
                return True
            self.sleep(0.5)
        return False

    def _click_template(self, frame, spec: TemplateSpec) -> bool:
        result = self._match(frame, spec)
        if not self._passes(result, spec):
            return False
        height, width = frame.shape[:2]
        x, y = result.center
        self.operate_click(
            max(0.0, min(1.0, x / width)),
            max(0.0, min(1.0, y / height)),
            name=spec.name,
            after_sleep=0.2,
        )
        return True

    def _advance_stage(self, stage: int) -> str:
        """Advance one EP. Returns ``ended`` / ``idle`` / ``limit`` / ``failed``.

        The chapter separator "EPn. 对话结束" has the highest priority: once it
        appears we stop clicking immediately, because the game only unlocks the
        next episode after leaving GLU TALK and re-opening it from the home screen.
        """
        self.info_set("对话阶段", stage)
        self.info_set("当前阶段", f"第 {stage} 段对话推进中")
        if not hasattr(self, "_click_total"):
            self._click_total = 0
        timeout = max(10.0, float(self.config.get("每段对话最长秒数", 90.0)))
        stage_clicks_limit = max(1, min(60, int(self.config.get("每段最多点击次数", 12))))
        total_clicks_limit = max(1, min(200, int(self.config.get("最多推进点击次数", 40))))
        idle_limit = max(1, int(self.config.get("无动作判定轮数", 6)))
        interval = max(0.2, float(self.config.get("回复点击间隔秒数", 0.8)))
        end_at = monotonic() + timeout
        clicks = 0
        idle = 0

        while (
            monotonic() < end_at
            and clicks < stage_clicks_limit
            and self._click_total < total_clicks_limit
        ):
            frame, boxes, text = self._read()
            height, width = frame.shape[:2]
            if dialogue_ended(text):
                episode = episode_number(text)
                previous = getattr(self, "_last_episode", None)
                if (
                    stage > 1 and episode is not None
                    and previous is not None and episode <= previous
                ):
                    self.info_set("状态", "重进后仍是上一段对话记录，没有新阶段。")
                    return "idle"
                self._last_episode = episode
                self.info_set("状态", f"第 {stage} 段对话结束，停止点击。")
                return "ended"
            view = classify_talk_page(text)
            self.info_set("页面判断", view.page)
            self._diagnose(view.diagnostics)
            remaining, unread = self._report_counters(text)
            action: TalkAction | None = None

            if view.page == "popup":
                action = dialogue_decision(
                    text, boxes, width, height,
                    allow_skip=False,
                    allow_choice=bool(self.config.get("点击对话选项", False)),
                )
            elif view.page == "dialogue":
                if bool(self.config.get("使用跳过按钮", True)):
                    if self._click_template(frame, SKIP_TEMPLATE) or self._click_template(
                        frame, TALK_TEMPLATE,
                    ):
                        clicks += 1
                        self._click_total += 1
                        idle = 0
                        self.info_set("推进点击数", self._click_total)
                        self.sleep(interval)
                        continue
                action = dialogue_decision(
                    text, boxes, width, height,
                    allow_skip=bool(self.config.get("使用跳过按钮", True)),
                    allow_choice=bool(self.config.get("点击对话选项", False)),
                )
            elif view.page == "list":
                if bool(self.config.get("自动发送回复", True)):
                    reply, evidence = reply_button_point(boxes, width, height)
                    self.info_set("回复证据", evidence)
                    if reply is not None:
                        action = TalkAction("reply", reply, evidence)
                if action is None:
                    badge, badge_evidence = unread_badge_point(
                        boxes, width, height, unread=unread,
                    )
                    self.info_set("可对话证据", badge_evidence)
                    if badge is not None:
                        action = TalkAction("conversation", badge, badge_evidence)
                    else:
                        target, evidence = available_task_point(
                            boxes, width, height, self._keywords(),
                        )
                        if target is not None and unread != 0:
                            self.info_set("可对话证据", evidence)
                            action = TalkAction("conversation", target, evidence)
            else:
                idle += 1
                if idle >= idle_limit:
                    self._fail("页面判定持续未知")
                    return "failed"
                self.sleep(0.5)
                continue

            if action is None:
                if remaining == 0 and (unread == 0 or unread is None):
                    self.info_set("状态", "今日剩余相遇 0，且没有未读对话。")
                    return "idle"
                idle += 1
                if idle >= idle_limit:
                    self.info_set("状态", "没有可对话的任务。")
                    return "idle"
                self.sleep(0.6)
                continue

            self.operate_click(
                *action.point, name=f"格鲁TALK{action.kind}", after_sleep=0.3,
            )
            clicks += 1
            self._click_total += 1
            idle = 0
            self.info_set("推进点击数", self._click_total)
            self.sleep(interval)

        if clicks >= stage_clicks_limit:
            self.log_warning("格鲁TALK：本段对话达到点击上限，安全停止。")
        return "limit"

    def _restart_from_home(self) -> bool:
        """Close GLU TALK, confirm the home screen, then open GLU TALK again."""
        self.info_set("当前阶段", "对话结束：返回主界面并重新打开格鲁TALK")
        frame, boxes, _ = self._read()
        height, width = frame.shape[:2]
        close = close_button_point(boxes, width, height)
        if close is not None:
            self.operate_click(*close, name="格鲁TALK关闭", after_sleep=0.6)
        else:
            self.try_return_home_with_back(max_presses=1)
        # Wait until the GLU TALK window is really gone before going home.
        end_at = monotonic() + 5.0
        while monotonic() < end_at:
            _, _, text = self._read()
            if classify_talk_page(text).page != "list":
                break
            self.sleep(0.4)
        else:
            self.try_return_home_with_back(max_presses=2)
        if not self._safe_return_home():
            return False
        if not self._wait_for_home_confirmation("重进格鲁TALK前主页确认"):
            return False
        frame, boxes, _ = self._read()
        entry = home_talk_entry(boxes, frame.shape[1], frame.shape[0])
        if entry is None:
            return False
        self.operate_click(*entry, name="格鲁TALK入口", after_sleep=1.0)
        self.sleep(float(self.config.get("重进后等待秒数", 2.0)))
        return self._wait_for_page(
            "list", float(self.config.get("页面等待秒数", 12.0)),
        )

    def _has_new_stage(
        self, text, boxes, width: int, height: int, previous_episode: int | None,
    ) -> bool:
        """Only a NEW episode/unread/reply counts as a next stage."""
        episode = episode_number(text)
        if dialogue_ended(text) and (
            episode is None or previous_episode is None or episode <= previous_episode
        ):
            # The old "EP2.对话结束" line stays in the chat history after re-entry.
            return False
        reply, _ = reply_button_point(boxes, width, height)
        if reply is not None:
            return True
        if (unread_count(text) or 0) > 0:
            return True
        if (remaining_encounters(text) or 0) > 0:
            return True
        return bool(
            episode is not None
            and previous_episode is not None
            and episode > previous_episode
        )

    def _scan(self) -> bool:
        """Episode loop: one EP ends -> back to home -> reopen GLU TALK."""
        limit = max(1, min(10, int(self.config.get("最多对话次数", 3))))
        self._click_total = 0
        self._last_episode = None
        self.info_set("完成对话数", 0)
        self.info_set("推进点击数", 0)
        completed = 0

        for stage in range(1, limit + 1):
            result = self._advance_stage(stage)
            if result == "failed":
                return False
            if result == "ended":
                completed += 1
                self.info_set("完成对话数", completed)
            if result != "ended" or stage >= limit:
                break
            if not bool(self.config.get("每段结束后回主页重进", True)):
                break
            if not self._restart_from_home():
                return self._fail("对话结束后未能回到主界面并重进格鲁TALK")
            frame, boxes, text = self._read()
            if not self._has_new_stage(
                text, boxes, frame.shape[1], frame.shape[0], self._last_episode,
            ):
                self.info_set("状态", "没有下一阶段对话。")
                break
        return True

    def run(self):
        if not bool(self.config.get("启用", True)):
            self.info_set("状态", "格鲁TALK 已禁用。")
            return True
        if bool(self.config.get("仅探测模式", False)):
            return self._probe()

        self.info_set("状态", "开始检查格鲁TALK 可对话会话。")
        frame, _, text = self._read()
        view = classify_talk_page(text)
        self.info_set("页面判断", view.page)
        self._diagnose(view.diagnostics)
        self._report_counters(text)
        if view.page == "list":
            # Already inside GLU TALK: do not press back or re-click the entry.
            self.info_set("当前阶段", "已在格鲁TALK页面，直接检查")
        else:
            self.info_set("当前阶段", "主页确认")
            if not self._wait_for_home_confirmation("格鲁TALK入口前主页确认"):
                if not self.try_return_home_with_back(
                    max_presses=int(self.config.get("回主页返回键最多次数", 3)),
                ):
                    return self._fail("未确认主页")
            frame, boxes, _ = self._read()
            entry = home_talk_entry(boxes, frame.shape[1], frame.shape[0])
            if entry is None:
                return self._fail("主页左列未识别到格鲁TALK入口")
            self.info_set("当前阶段", "进入格鲁TALK")
            self.operate_click(*entry, name="格鲁TALK入口", after_sleep=1.0)
            if not self._wait_for_page(
                "list", float(self.config.get("页面等待秒数", 12.0)),
            ):
                return self._fail("未确认进入格鲁TALK页面")

        if not self._scan():
            return False
        self._safe_return_home()
        info = getattr(self, "info", {}) or {}
        completed = info.get("完成对话数", 0)
        clicks = info.get("推进点击数", 0)
        self.info_set("状态", "格鲁TALK 检查完成。")
        self.log_completion(
            f"格鲁TALK 检查完成：点开 {completed} 个会话、推进 {clicks} 次，已返回主页。"
        )
        return True
