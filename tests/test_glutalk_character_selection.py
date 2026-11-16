"""Regression for the empty GLU TALK pane and the 新的缘分 character list."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image

from src.tasks.glutalk_vision import (
    CONVERSATION_TAB_BADGE_ROI,
    available_task_point,
    conversation_tab_point,
    new_fate_character_point,
    remaining_encounters,
    reply_button_point,
    unavailable_contact,
)
from tests.test_glutalk import make_probe_task


def label(name, x, y, width=3840, height=2160):
    return SimpleNamespace(
        name=name,
        x=(x - 0.05) * width,
        y=(y - 0.01) * height,
        width=0.1 * width,
        height=0.02 * height,
        confidence=0.99,
    )


def character_list(width=3840, height=2160):
    return [
        label(name, x, y, width, height)
        for name, x, y in (
            ("GLU TALK", 0.28, 0.25),
            ("今日的剩余相遇|3", 0.70, 0.25),
            ("缘分列表4", 0.30, 0.40),
            ("新的缘分1", 0.30, 0.45),
            ("被封印的欲望 阿奇拉", 0.36, 0.51),
            ("缘分4", 0.28, 0.58),
            ("新进员工 内布利斯", 0.36, 0.63),
            ("MAX", 0.45, 0.63),
            ("开始对话吧", 0.66, 0.60),
        )
    ]


def unread_tab_frame():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    path = Path(__file__).parent / "fixtures" / "glutalk_unread_tab.png"
    crop = np.array(Image.open(path).convert("RGB"))[..., ::-1]
    x = int(CONVERSATION_TAB_BADGE_ROI[0] * 1920)
    y = int(CONVERSATION_TAB_BADGE_ROI[1] * 1080)
    frame[y : y + crop.shape[0], x : x + crop.shape[1]] = crop
    return frame


class CharacterSelectionTest(unittest.TestCase):
    def test_real_unread_badge_opens_chat_tab_instead_of_unowned_contact(self):
        boxes = character_list(1920, 1080)
        point, _ = conversation_tab_point(unread_tab_frame(), boxes)
        self.assertIsNotNone(point)
        self.assertTrue(0.21 < point[0] < 0.25)
        self.assertTrue(0.36 < point[1] < 0.45)
        self.assertIsNone(conversation_tab_point(unread_tab_frame(), [])[0])
        self.assertIsNone(conversation_tab_point(np.zeros((1080, 1920, 3), np.uint8), boxes)[0])

    def test_unowned_character_and_costume_info_are_not_dialogue_evidence(self):
        self.assertTrue(unavailable_contact("未拥有 持有该服装时，可获得联系方式。"))
        self.assertFalse(unavailable_contact("GLU TALK 今日的剩余相遇3 你好"))
        self.assertIsNone(reply_button_point([label("服装信息", 0.66, 0.75)], 3840, 2160)[0])

    def test_stage_prioritizes_unread_tab_then_the_unread_character(self):
        task = make_probe_task(仅探测模式=False)
        task.operate_click = Mock()
        frame = unread_tab_frame()
        boxes = character_list(1920, 1080)
        unread_boxes = [label("1", 0.49, 0.55, 1920, 1080)]
        task._read = Mock(
            side_effect=[
                (frame, boxes, " ".join(box.name for box in boxes) + " 未拥有 服装信息"),
                (frame, unread_boxes, "GLU TALK 今日的剩余相遇3 未读消息1"),
                (
                    frame,
                    [
                        label("你好", 0.66, 0.75, 1920, 1080),
                        label("发送消息", 0.56, 0.70, 1920, 1080),
                    ],
                    "GLU TALK 今日的剩余相遇3",
                ),
                (frame, [], "GLU TALK EP1.对话结束"),
            ]
        )
        self.assertEqual("ended", task._advance_stage(1))
        self.assertEqual(
            ["格鲁TALKconversation_tab", "格鲁TALKconversation", "格鲁TALKreply"],
            [call.kwargs["name"] for call in task.operate_click.call_args_list],
        )

    def test_old_episode_in_history_does_not_hide_a_new_reply(self):
        task = make_probe_task(仅探测模式=False)
        task.operate_click = Mock()
        frame = np.zeros((2160, 3840, 3), np.uint8)
        boxes = [label("保证什么？", 0.66, 0.75), label("发送消息", 0.56, 0.70)]
        text = "GLU TALK 今日的剩余相遇3 EP2.对话结束 你保证过的，对吧？ 保证什么？"
        self.assertTrue(task._has_new_stage(text, boxes, 3840, 2160, 2))
        task._read = Mock(
            side_effect=[
                (frame, boxes, text),
                (frame, [], "GLU TALK EP3.对话结束"),
            ]
        )
        self.assertEqual("ended", task._advance_stage(1))
        task.operate_click.assert_called_once()
        self.assertEqual("格鲁TALKreply", task.operate_click.call_args.kwargs["name"])
        self.assertEqual(3, task._last_episode)

    def test_real_screenshot_selects_aquila_instead_of_empty_pane(self):
        path = Path(__file__).parent / "fixtures" / "glutalk_character_list.json"
        fixture = json.loads(path.read_text(encoding="utf-8"))
        boxes = [SimpleNamespace(**box) for box in fixture["boxes"]]
        width, height = fixture["width"], fixture["height"]
        self.assertEqual(3, remaining_encounters(" ".join(box.name for box in boxes)))
        point, character = new_fate_character_point(boxes, width, height)
        self.assertIn("阿奇拉", character)
        self.assertAlmostEqual(0.358, point[0], delta=0.005)
        self.assertAlmostEqual(0.508, point[1], delta=0.005)
        self.assertIsNone(available_task_point(boxes, width, height)[0])

    def test_counter_accepts_separators_and_zero(self):
        for separator in ("", "|", "｜", "丨", ":", "："):
            with self.subTest(separator=separator):
                self.assertEqual(3, remaining_encounters(f"今日的剩余相遇 {separator} 3 +"))
                self.assertEqual(0, remaining_encounters(f"今日的剩余相遇 {separator} 0 +"))
        self.assertIsNone(remaining_encounters("今日的剩余相遇 +"))

    def test_empty_pane_instruction_is_never_a_click_target(self):
        self.assertIsNone(available_task_point(character_list(), 3840, 2160)[0])
        self.assertIsNone(reply_button_point([label("开始对话吧", 0.66, 0.75)], 3840, 2160)[0])

    def test_selects_new_character_at_multiple_resolutions(self):
        for width, height in ((1920, 1080), (2560, 1440), (3840, 2160)):
            with self.subTest(resolution=(width, height)):
                point, character = new_fate_character_point(
                    character_list(width, height),
                    width,
                    height,
                )
                self.assertAlmostEqual(0.36, point[0])
                self.assertAlmostEqual(0.51, point[1])
                self.assertIn("阿奇拉", character)

    def test_missing_or_empty_new_section_never_selects_old_max_characters(self):
        for replacement in ("新的缘分0", "缘分列表4", ""):
            boxes = character_list()
            boxes[3].name = replacement
            self.assertIsNone(new_fate_character_point(boxes, 3840, 2160)[0])
        boxes = character_list()
        boxes.pop(4)
        self.assertIsNone(new_fate_character_point(boxes, 3840, 2160)[0])

    def test_excludes_tried_characters_and_advances_to_next(self):
        boxes = character_list()
        boxes[3].name = "新的缘分2"
        boxes.append(label("另一名角色", 0.36, 0.55))
        _, first = new_fate_character_point(boxes, 3840, 2160)
        point, second = new_fate_character_point(boxes, 3840, 2160, excluded={first})
        self.assertEqual("另一名角色", second)
        self.assertAlmostEqual(0.55, point[1])
        self.assertIsNone(
            new_fate_character_point(
                boxes,
                3840,
                2160,
                excluded={first, second},
            )[0]
        )

    def test_max_marker_on_same_row_disqualifies_character(self):
        boxes = character_list() + [label("MAX", 0.46, 0.51)]
        self.assertIsNone(new_fate_character_point(boxes, 3840, 2160)[0])

    def test_stage_selects_character_then_replies_and_confirms_end(self):
        task = make_probe_task(仅探测模式=False)
        task.operate_click = Mock()
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        boxes = character_list()
        text = " ".join(box.name for box in boxes)
        task._read = Mock(
            side_effect=[
                (frame, boxes, text),
                (
                    frame,
                    [label("你好", 0.66, 0.75), label("发送消息", 0.56, 0.70)],
                    "GLU TALK 今日的剩余相遇|2 你好",
                ),
                (frame, [], "GLU TALK EP1.对话结束"),
            ]
        )
        self.assertEqual("ended", task._advance_stage(1))
        self.assertEqual(2, task.operate_click.call_count)
        self.assertEqual("格鲁TALKcharacter", task.operate_click.call_args_list[0].kwargs["name"])
        self.assertEqual("格鲁TALKreply", task.operate_click.call_args_list[1].kwargs["name"])

    def test_unresponsive_character_is_bounded_and_not_reported_complete(self):
        task = make_probe_task(仅探测模式=False)
        task.operate_click = Mock()
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        boxes = character_list()
        task._read = Mock(return_value=(frame, boxes, " ".join(box.name for box in boxes)))
        self.assertEqual("failed", task._advance_stage(1))
        self.assertEqual(2, task.operate_click.call_count)
        self.assertIn("未确认", task.info["状态"])

    def test_new_encounters_are_not_started_without_confirmed_remaining_count(self):
        for counter in ("今日的剩余相遇|0", "今日的剩余相遇"):
            with self.subTest(counter=counter):
                task = make_probe_task(仅探测模式=False)
                task.operate_click = Mock()
                boxes = character_list()
                boxes[1].name = counter
                frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
                task._read = Mock(return_value=(frame, boxes, " ".join(b.name for b in boxes)))
                self.assertEqual("idle", task._advance_stage(1))
                task.operate_click.assert_not_called()
