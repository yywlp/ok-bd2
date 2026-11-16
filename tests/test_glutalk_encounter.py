"""Regressions for unread badges competing with the current encounter card."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image

from src.tasks.glutalk_vision import (
    classify_talk_page,
    dialogue_decision,
    encounter_button_point,
)
from tests.test_glutalk import make_probe_task
from tests.test_glutalk_character_selection import label


def encounter_boxes(button="进行相遇事件"):
    return [
        label(name, x, y)
        for name, x, y in (
            ("GLU TALK", 0.29, 0.25),
            ("今日的剩余相遇|3", 0.70, 0.25),
            ("未读消息1", 0.30, 0.32),
            ("1", 0.508, 0.393),
            ("相遇事件", 0.61, 0.62),
            ("第3次相遇", 0.61, 0.67),
            (button, 0.66, 0.753),
        )
    ]


class EncounterTest(unittest.TestCase):
    def test_encounter_replay_hint_in_chat_history_does_not_become_a_popup(self):
        text = (
            "GLU TALK 今日的剩余相遇2 未读消息1 "
            "可通过格鲁TALK>简介>相遇事件，再次查看已完成的逸闻。 "
            "回放相遇事件 发送消息 我争取。"
        )
        self.assertEqual("list", classify_talk_page(text).page)

    def test_affinity_reward_requires_both_title_and_reward_description(self):
        boxes = [
            label("获得好感度奖励", 0.5, 0.36),
            label("获得代号O爱丽洁的好感度奖励。", 0.5, 0.69),
        ]
        text = " ".join(b.name for b in boxes)
        self.assertEqual("popup", classify_talk_page(text).page)
        self.assertEqual("dismiss_reward", dialogue_decision(text, boxes, 3840, 2160).kind)
        self.assertIsNone(dialogue_decision(boxes[0].name, boxes[:1], 3840, 2160))

    def test_skip_confirmation_clicks_confirm_instead_of_modal_title(self):
        boxes = [
            label(name, x, y)
            for name, x, y in (
                ("跳过对话", 0.5, 0.286),
                ("可通过格鲁TALK>简介>相遇事件，再次查看已完成的逸闻。", 0.5, 0.63),
                ("取消", 0.45, 0.724),
                ("确认", 0.545, 0.724),
            )
        ]
        text = " ".join(b.name for b in boxes)
        self.assertEqual("popup", classify_talk_page(text).page)
        action = dialogue_decision(text, boxes, 3840, 2160)
        self.assertEqual("skip_confirm", action.kind)
        self.assertEqual((0.545, 0.724), action.point)
        self.assertIsNone(dialogue_decision(text, boxes, 3840, 2160, allow_skip=False))
        self.assertIsNone(dialogue_decision(text, boxes[:-1], 3840, 2160))

    def test_icon_only_encounter_uses_real_skip_template_without_ocr_keyword(self):
        frame = np.zeros((1080, 1920, 3), np.uint8)
        crop = np.array(
            Image.open(Path(__file__).parent / "fixtures" / "glutalk_encounter_skip.png").convert(
                "RGB"
            )
        )[..., ::-1]
        frame[14:116, 1372:1770] = crop
        task = make_probe_task(仅探测模式=False)
        task.operate_click = Mock()
        task._read = Mock(
            side_effect=[
                (frame, [], "F1 F2 F3 好宁静啊。"),
                (frame, [], "GLU TALK EP3.对话结束"),
            ]
        )
        self.assertEqual("ended", task._advance_stage(1))
        task.operate_click.assert_called_once()
        self.assertEqual("格鲁TALK跳过", task.operate_click.call_args.kwargs["name"])

    def test_free_encounter_confirmation_does_not_require_story_choice_setting(self):
        boxes = [
            label(name, x, y)
            for name, x, y in (
                ("是否要进行相遇事件？", 0.50, 0.48),
                ("可进行次数：3次", 0.50, 0.515),
                ("取消", 0.45, 0.633),
                ("确认", 0.55, 0.633),
            )
        ]
        text = " ".join(b.name for b in boxes)
        self.assertEqual("popup", classify_talk_page(text).page)
        action = dialogue_decision(text, boxes, 3840, 2160, allow_choice=False)
        self.assertEqual("encounter_confirm", action.kind)
        self.assertEqual((0.55, 0.633), action.point)
        for counter in ("可进行次数：0次", "可进行次数", "购买次数3次"):
            boxes[1].name = counter
            self.assertIsNone(
                dialogue_decision(
                    " ".join(b.name for b in boxes), boxes, 3840, 2160, allow_choice=True
                )
            )

    def test_confirmation_cannot_fall_back_to_a_background_or_purchase_button(self):
        boxes = [
            label("是否要进行相遇事件？", 0.5, 0.48),
            label("可进行次数：3次", 0.5, 0.515),
            label("确认", 0.66, 0.75),
        ]
        self.assertIsNone(
            dialogue_decision(" ".join(b.name for b in boxes), boxes, 3840, 2160, allow_choice=True)
        )

    def test_real_ocr_from_report_and_live_frame_identifies_encounter(self):
        fixture = json.loads(
            (Path(__file__).parent / "fixtures" / "glutalk_encounter.json").read_text(
                encoding="utf-8"
            )
        )
        for sample in fixture:
            with self.subTest(sample=sample["name"]):
                boxes = [SimpleNamespace(**b) for b in sample["boxes"]]
                point, _ = encounter_button_point(boxes, sample["width"], sample["height"])
                self.assertIsNotNone(point)
                self.assertAlmostEqual(0.664, point[0], delta=0.003)
                self.assertAlmostEqual(0.753, point[1], delta=0.003)

    def test_affinity_icon_digits_are_allowed_but_replay_is_never_started(self):
        for name in ("进行相遇事件", "3进行相遇事件", "③进行相遇事件", "💔进行相遇事件"):
            with self.subTest(button=name):
                self.assertIsNotNone(encounter_button_point(encounter_boxes(name), 3840, 2160)[0])
        for name in ("回放相遇事件", "回放进行相遇事件", "不可进行相遇事件"):
            self.assertIsNone(encounter_button_point(encounter_boxes(name), 3840, 2160)[0])

    def test_missing_card_or_ambiguous_buttons_are_not_clicked(self):
        boxes = encounter_boxes()
        self.assertIsNone(encounter_button_point(boxes[:4] + boxes[-1:], 3840, 2160)[0])
        self.assertIsNone(
            encounter_button_point(boxes + [label("进行相遇事件", 0.66, 0.79)], 3840, 2160)[0]
        )

    def test_encounter_precedes_unread_badge_and_old_episode_history(self):
        task = make_probe_task(仅探测模式=False)
        task.operate_click = Mock()
        frame = np.zeros((2160, 3840, 3), np.uint8)
        boxes = encounter_boxes("③进行相遇事件")
        text = " ".join(b.name for b in boxes) + " EP2.对话结束"
        self.assertTrue(task._has_new_stage(text, boxes, 3840, 2160, 2))
        task._read = Mock(
            side_effect=[
                (frame, boxes, text),
                (
                    frame,
                    [
                        label("是否要进行相遇事件？", 0.5, 0.48),
                        label("可进行次数：3次", 0.5, 0.515),
                        label("确认", 0.55, 0.633),
                    ],
                    "是否要进行相遇事件？可进行次数：3次 确认",
                ),
                (
                    frame,
                    [label("你好", 0.66, 0.75), label("发送消息", 0.56, 0.70)],
                    "GLU TALK 今日的剩余相遇|2 你好",
                ),
                (frame, [], "GLU TALK EP3.对话结束"),
            ]
        )
        self.assertEqual("ended", task._advance_stage(1))
        self.assertEqual(
            ["格鲁TALKencounter", "格鲁TALKencounter_confirm", "格鲁TALKreply"],
            [c.kwargs["name"] for c in task.operate_click.call_args_list],
        )

    def test_unresponsive_encounter_stops_without_reselecting_unread_role(self):
        task = make_probe_task(仅探测模式=False)
        task.operate_click = Mock()
        boxes = encounter_boxes()
        task._read = Mock(
            return_value=(
                np.zeros((2160, 3840, 3), np.uint8),
                boxes,
                " ".join(b.name for b in boxes),
            )
        )
        self.assertEqual("failed", task._advance_stage(1))
        self.assertEqual(
            ["格鲁TALKencounter"] * 2, [c.kwargs["name"] for c in task.operate_click.call_args_list]
        )
        self.assertIn("相遇条件", task.info["状态"])

    def test_encounter_requires_positive_remaining_quota(self):
        for counter in ("今日的剩余相遇|0", "今日的剩余相遇"):
            with self.subTest(counter=counter):
                task = make_probe_task(仅探测模式=False)
                task.operate_click = Mock()
                boxes = encounter_boxes()
                boxes[1].name = counter
                task._read = Mock(
                    return_value=(
                        np.zeros((2160, 3840, 3), np.uint8),
                        boxes,
                        " ".join(b.name for b in boxes),
                    )
                )
                self.assertEqual("failed", task._advance_stage(1))
                task.operate_click.assert_not_called()

    def test_unresponsive_unread_badge_is_only_selected_twice(self):
        task = make_probe_task(仅探测模式=False)
        task.operate_click = Mock()
        boxes = encounter_boxes()[:4]
        task._read = Mock(
            return_value=(
                np.zeros((2160, 3840, 3), np.uint8),
                boxes,
                " ".join(b.name for b in boxes),
            )
        )
        self.assertEqual("failed", task._advance_stage(1))
        self.assertEqual(
            ["格鲁TALKconversation"] * 2,
            [c.kwargs["name"] for c in task.operate_click.call_args_list],
        )
