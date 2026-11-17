from src.icons import TaskIcon as FluentIcon
from src.tasks.BaseBD2Task import BaseBD2Task
from src.tasks.main_story import (
    MainStoryConfigMixin,
    MainStoryFeatureMixin,
)
from src.tasks.task_vision_mixin import TaskVisionMixin


class MainStoryFarmTask(
    MainStoryFeatureMixin,
    MainStoryConfigMixin,
    TaskVisionMixin,
    BaseBD2Task,
):
    """主线「快速探索」循环刷取。

    与快速狩猎不同，这里刷的是主线卡带里已解锁的高难度关卡：进关卡后点
    「进行快速探索」，游戏会自动推进节点、遇战斗自动接手，本任务只负责在
    正确的时候点「自动进行 / 战斗 / 离开」，并在认不出界面时停下来存证。
    """

    status_keys = [
        "启用",
        "状态",
        "主线当前界面",
        "主线关卡标题",
        "主线战斗次数",
        "主线自动进行次数",
        "主线完成轮次",
        "主线停止原因",
        "主线快速探索入口 OCR错误",
        "主线列表按钮 OCR错误",
        "主线结算离开 OCR错误",
        "主线结算标题 OCR错误",
        "主线关卡标题 OCR错误",
        "主线截图错误",
        "截图文件",
        "鼠标点击",
        "Log",
        "Warning",
        "Error",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_vision_state()
        self._install_main_story_config()
        self.visible = True
        self.group_name = "自动刷级"
        self.group_icon = FluentIcon.CALENDAR
        self.name = "主线快速探索"
        self.description = (
            "在已通关的主线高难关卡里重复「进行快速探索」，自动点自动进行、"
            "战斗与离开，直到达到战斗次数或时间上限。请先把角色停在该关卡内再开始。"
        )
        self.icon = FluentIcon.GAME

        visible_keys = ["启用", *MainStoryConfigMixin.MAIN_STORY_CONFIG_KEYS]
        self.config_type["启用"] = {"sub_configs": {True: visible_keys}}
        ordered = {
            key: self.default_config[key]
            for key in visible_keys
            if key in self.default_config
        }
        ordered.update(
            {
                key: value
                for key, value in self.default_config.items()
                if key not in ordered
            }
        )
        self.default_config = ordered

    def run(self):
        if not bool(self.config.get("启用", True)):
            self.info_set("状态", "主线快速探索已禁用。")
            self.log_info("主线快速探索已禁用。")
            return True

        self.info_set("状态", "主线快速探索启动。")
        success = self.run_main_story_farm()
        self.info_set("状态", "主线快速探索完成。" if success else "主线快速探索停止。")
        if success:
            self.log_completion("主线快速探索：已到达设定的停止条件。")
        return success
