import unittest

from src.tasks.main_story import (
    MainStoryState,
    classify_main_story_state,
)


class ClassifyMainStoryStateTest(unittest.TestCase):

    def test_victory_on_leave_button(self):
        self.assertIs(
            classify_main_story_state("", "", "离开", ""),
            MainStoryState.VICTORY,
        )

    def test_victory_tolerates_spaced_ocr(self):
        self.assertIs(
            classify_main_story_state("", "", "离 开", ""),
            MainStoryState.VICTORY,
        )

    def test_battle_button_wins_over_empty_leave_roi(self):
        # 列表页这两块 ROI 在画面上重叠，读不到「离开」时必须走列表分支。
        self.assertIs(
            classify_main_story_state("", "战斗", "", ""),
            MainStoryState.LIST_BATTLE,
        )

    def test_in_stage_entry(self):
        self.assertIs(
            classify_main_story_state("进行快速探索", "", "", ""),
            MainStoryState.IN_STAGE,
        )

    def test_list_auto(self):
        self.assertIs(
            classify_main_story_state("", "自动进行", "", ""),
            MainStoryState.LIST_AUTO,
        )

    def test_battle_keyword_not_swallowed_by_auto_text(self):
        # 「自动进行」里不含「战斗」，两者不能互相误判。
        self.assertIs(
            classify_main_story_state("", "自动进行", "", ""),
            MainStoryState.LIST_AUTO,
        )

    def test_unknown_button_text(self):
        self.assertIs(
            classify_main_story_state("", "领取奖励", "", ""),
            MainStoryState.LIST_OTHER_BUTTON,
        )

    def test_empty_everything_is_busy(self):
        self.assertIs(classify_main_story_state("", "", "", ""), MainStoryState.BUSY)

    def test_ocr_noise_still_classifies(self):
        # 实战里「进」常被识别成形近字，允许一字之差。
        self.assertIs(
            classify_main_story_state("", "自动逬行", "", ""),
            MainStoryState.LIST_AUTO,
        )

    def test_whitespace_is_ignored(self):
        self.assertIs(
            classify_main_story_state(" 进行 快速 探索 ", "", "", ""),
            MainStoryState.IN_STAGE,
        )

    def test_values_are_stable_strings(self):
        # 状态值会写进 GUI 状态面板，改名属于破坏性改动。
        self.assertEqual(MainStoryState.IN_STAGE.value, "关卡内")
        self.assertEqual(MainStoryState.LIST_BATTLE.value, "列表-战斗")
        self.assertEqual(MainStoryState.VICTORY.value, "战斗结算")

    # --- 2026-11-16 实机失败回归：战斗内 HUD 落在列表按钮 ROI 里 ---

    def test_battle_hud_autobattle_hint_is_not_battle_button(self):
        # 实机日志：按钮 ROI 读到「结束自动战斗esc」，含「战斗」二字但不是按钮。
        self.assertIs(
            classify_main_story_state("", "结束自动战斗esc", "结束自动战斗esc", ""),
            MainStoryState.IN_BATTLE,
        )

    def test_battle_hud_turn_battle_is_not_unknown_button(self):
        # 实机导致停机的那一帧：按钮=turn.battle，离开=turn.1」battle。
        self.assertIs(
            classify_main_story_state("", "turn.battle", "turn.1」battle", ""),
            MainStoryState.IN_BATTLE,
        )

    def test_battle_hud_turn_counter_only(self):
        self.assertIs(
            classify_main_story_state("", "TURN.1BATTLE", "", ""),
            MainStoryState.IN_BATTLE,
        )

    def test_battle_hud_does_not_swallow_list_battle(self):
        # 列表页那个按钮是中文「战斗」，不能被 HUD 规则吃掉。
        self.assertIs(
            classify_main_story_state("", "战斗", "", ""),
            MainStoryState.LIST_BATTLE,
        )

    def test_battle_hud_does_not_swallow_auto(self):
        self.assertIs(
            classify_main_story_state("", "自动进行", "", ""),
            MainStoryState.LIST_AUTO,
        )

    def test_battle_hud_does_not_swallow_victory(self):
        self.assertIs(
            classify_main_story_state("", "", "离开", ""),
            MainStoryState.VICTORY,
        )

    def test_in_battle_value_is_stable(self):
        self.assertEqual(MainStoryState.IN_BATTLE.value, "战斗内")

    # --- 2026-11-16 奖励弹窗回归 ---

    def test_reward_popup_detected(self):
        self.assertIs(
            classify_main_story_state("", "", "", "可获得奖励 点击画面即可返回"),
            MainStoryState.REWARD_POPUP,
        )

    def test_list_reward_preview_label_is_not_a_popup(self):
        # 2026-11-17 实机：列表页中部常驻「可获得奖励」奖励预览标签，位置和弹窗
        # 标题 ROI 完全重叠；只认这段文字会把整张列表页当弹窗、点空地到超时。
        self.assertIs(
            classify_main_story_state("", "自动进行", "自动进行", "可获得奖励"),
            MainStoryState.LIST_AUTO,
        )

    def test_reward_popup_wins_over_auto_button(self):
        # 真弹窗帧：底部提示「点击画面即可返回」+ 背后仍可见的「自动进行」。
        self.assertIs(
            classify_main_story_state(
                "", "自动进行", "自动进行", "可获得奖励 点击画面即可返回",
            ),
            MainStoryState.REWARD_POPUP,
        )

    def test_reward_popup_hint_without_reward_title(self):
        self.assertIs(
            classify_main_story_state("", "", "", "点击画面即可返回"),
            MainStoryState.REWARD_POPUP,
        )

    def test_reward_popup_tolerates_one_wrong_character(self):
        self.assertIs(
            classify_main_story_state("", "", "", "点击画面即可返凹"),
            MainStoryState.REWARD_POPUP,
        )

    def test_reward_popup_does_not_swallow_battle(self):
        self.assertIs(
            classify_main_story_state("", "战斗", "战斗", ""),
            MainStoryState.LIST_BATTLE,
        )

    def test_reward_popup_does_not_swallow_victory(self):
        self.assertIs(
            classify_main_story_state("", "", "离开", ""),
            MainStoryState.VICTORY,
        )

    def test_reward_popup_does_not_swallow_in_stage(self):
        self.assertIs(
            classify_main_story_state("进行快速探索", "", "", ""),
            MainStoryState.IN_STAGE,
        )

    def test_reward_popup_value_is_stable(self):
        self.assertEqual(MainStoryState.REWARD_POPUP.value, "奖励弹窗")


if __name__ == "__main__":
    unittest.main()
