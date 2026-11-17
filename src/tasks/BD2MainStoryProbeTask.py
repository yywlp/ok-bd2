"""主线「快速探索」流程录制探针。

主线快速探索的界面帧尚未标定：入口按钮在关卡卡片还是关卡内、次数用 MAX 还是 +/−、
难度标签出现在地图标题还是顶部横条，这些都不能靠猜。本任务只做读取与落盘，不点击任何
按钮——用户在游戏里手动走一遍流程，任务按固定间隔连续截图并做全屏 OCR，
产出「帧 + 文本 + 坐标」三件套，供后续标定 ROI、OCR 词表和模板区域。
"""

from datetime import datetime
from time import monotonic

from src.icons import TaskIcon as FluentIcon
from src.tasks.BaseBD2Task import BaseBD2Task


class BD2MainStoryProbeTask(BaseBD2Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "BD2 主线流程探针"
        self.description = (
            "连续截图并做全屏 OCR，录制主线快速探索流程的界面帧，用于坐标与识别词表标定。"
        )
        self.icon = FluentIcon.SEARCH
        self.group_name = "测试"
        self.group_icon = FluentIcon.BOOK_SHELF
        self.visible = True
        self.default_config.update(
            {
                "录制时长秒数": 120.0,
                "采样间隔秒数": 1.5,
                "OCR 识别阈值": 0.2,
                "保存截图": True,
            }
        )
        self.config_description.update(
            {
                "录制时长秒数": "开始录制后持续采集的时间，到点自动结束。",
                "采样间隔秒数": "两帧之间的等待时间；OCR 本身耗时会让实际间隔略长。",
                "OCR 识别阈值": "全屏 OCR 使用的最低可信度。",
                "保存截图": "同时保存 PNG，关闭则只记录文本与坐标。",
            }
        )
        self.config_type.update(
            {
                "录制时长秒数": {"min": 10.0, "max": 900.0, "step": 10.0},
                "采样间隔秒数": {"min": 0.5, "max": 10.0, "step": 0.5},
                "OCR 识别阈值": {"min": 0.05, "max": 0.95, "step": 0.01},
            }
        )

    def run(self):
        duration = float(self.config.get("录制时长秒数", 120.0))
        interval = float(self.config.get("采样间隔秒数", 1.5))
        threshold = float(self.config.get("OCR 识别阈值", 0.2))
        save_shot = bool(self.config.get("保存截图", True))

        started = datetime.now()
        deadline = monotonic() + duration
        lines = [
            f"start={started:%Y-%m-%d %H:%M:%S}",
            f"capture_method={self.capture_method_name}",
            f"duration={duration:.0f}s interval={interval:.1f}s threshold={threshold:.2f}",
            "",
        ]

        index = 0
        failures = 0
        while monotonic() < deadline:
            index += 1
            stamp = datetime.now().strftime("%H%M%S")
            try:
                frame = self.capture_frame()
            except Exception as exc:  # 单帧失败不该中断整段录制
                failures += 1
                lines.append(f"# {index:03d} {stamp} capture_error={exc}")
                self.info_set("录制进度", f"{index} 帧（{failures} 次失败）")
                self.sleep(interval)
                continue

            height, width = frame.shape[:2]
            stem = f"main_story_{index:03d}_{stamp}"
            if save_shot:
                self.save_frame(stem, frame)

            boxes = self.ocr(frame=frame, threshold=threshold, log=False)
            lines.append(f"# {index:03d} {stamp} {width}x{height} text={len(boxes)}")
            for box in boxes:
                lines.append(f"  {box.name}\t{float(box.confidence):.3f}\t{box}")

            self.info_set("录制进度", f"{index} 帧 / 剩余 {max(0.0, deadline - monotonic()):.0f}s")
            self.sleep(interval)

        lines.append("")
        lines.append(f"end={datetime.now():%Y-%m-%d %H:%M:%S} frames={index} failures={failures}")
        output = self.write_probe_text("main_story_probe_latest.txt", lines)
        self.log_completion(f"主线流程录制完成：{index} 帧，文本 {output}")
        return True
