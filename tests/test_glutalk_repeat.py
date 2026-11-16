"""Reopen chats after each episode, and distinguish limits from completion."""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from src.tasks.glutalk_vision import (
    classify_talk_page,
    dialogue_decision,
    remaining_encounters,
    reply_button_point,
    unread_count,
)
from tests.test_glutalk import make_probe_task
from tests.test_glutalk_character_selection import character_list, label


class RepeatTalkTest(unittest.TestCase):
    def test_single_character_reply_is_allowed_only_with_send_control(self):
        boxes = [label("发送消息", 0.56, 0.70, 1920, 1080), label("嗯？", 0.66, 0.753, 1920, 1080)]
        point, _ = reply_button_point(boxes, 1920, 1080, require_send_control=True)
        self.assertIsNotNone(point)
        self.assertAlmostEqual(.66, point[0])
        self.assertAlmostEqual(.753, point[1])
        self.assertIsNone(reply_button_point(boxes[1:], 1920, 1080, require_send_control=True)[0])

    def test_counter_does_not_merge_with_clock_or_sidebar_digits(self):
        self.assertEqual(0, remaining_encounters("今日的剩余相遇0 36:18 X"))
        self.assertEqual(1, unread_count("未读消息1 8 代号O爱丽洁"))
        self.assertEqual(3, remaining_encounters("今日的剩余相遇３✨"))
        self.assertIsNone(remaining_encounters("今日的剩余相遇36"))

    def test_original_paired_click_limits_are_upgraded_but_custom_limits_remain(self):
        for stage_limit, total_limit, expected in (
            (12, 40, (24, 80)),
            (1, 40, (1, 40)),
            (12, 60, (12, 60)),
        ):
            task = make_probe_task(
                仅探测模式=False, 每段最多点击次数=stage_limit, 最多推进点击次数=total_limit
            )
            task._advance_stage = Mock(return_value="idle")
            task._restart_from_home = Mock(return_value=True)
            task._wait_for_next_stage = Mock(return_value=False)
            self.assertTrue(task._scan())
            self.assertEqual(
                expected, (task.config["每段最多点击次数"], task.config["最多推进点击次数"])
            )

    def test_waits_for_send_control_instead_of_clicking_incoming_messages(self):
        task = make_probe_task(仅探测模式=False)
        task.operate_click = Mock()
        incoming = [label("…你最好别站窗边啊。", 0.646, 0.766, 1920, 1080)]
        ready = incoming + [
            label("发送消息", 0.56, 0.70, 1920, 1080),
            label("那个，爱丽洁？", 0.66, 0.753, 1920, 1080),
        ]
        text = "GLU TALK 今日的剩余相遇1 未读消息1 …你最好别站窗边啊。"
        task._read = Mock(
            side_effect=[
                (self.frame, incoming, text),
                (self.frame, incoming, text),
                (self.frame, ready, text + " 发送消息 那个，爱丽洁？"),
                (self.frame, [], "GLU TALK 今日的剩余相遇1 未读消息0 EP4.对话结束"),
            ]
        )
        self.assertEqual("ended", task._advance_stage(1))
        task.operate_click.assert_called_once()
        self.assertAlmostEqual(0.66, task.operate_click.call_args.args[0])
        self.assertAlmostEqual(0.753, task.operate_click.call_args.args[1])

    def setUp(self):
        self.frame = np.zeros((1080, 1920, 3), np.uint8)

    def test_real_unlock_screen_returns_via_back_template_then_continues_chat(self):
        frame = self.frame.copy()
        crop = np.array(
            Image.open(
                Path(__file__).parent / "fixtures" / "glutalk_unlock_back.png",
            ).convert("RGB")
        )[..., ::-1]
        frame[:130, :288] = crop
        text = "D获得好感度特效！ 主界面设置"
        self.assertEqual("unlock", classify_talk_page(text).page)
        task = make_probe_task(仅探测模式=False)
        task.operate_click = Mock()
        task._read = Mock(
            side_effect=[
                (frame, [], text),
                (
                    self.frame,
                    [
                        label("你好", 0.66, 0.75, 1920, 1080),
                        label("发送消息", 0.56, 0.70, 1920, 1080),
                    ],
                    "GLU TALK 今日的剩余相遇1 未读消息1 你好",
                ),
                (self.frame, [], "GLU TALK 今日的剩余相遇1 未读消息0 EP4.对话结束"),
            ]
        )
        self.assertEqual("ended", task._advance_stage(1))
        names = [c.kwargs["name"] for c in task.operate_click.call_args_list]
        self.assertEqual(["格鲁TALK特效解锁返回", "格鲁TALKreply"], names)

    def test_unlock_requires_both_page_labels_and_a_visible_back_control(self):
        self.assertNotEqual("unlock", classify_talk_page("获得好感度特效").page)
        self.assertNotEqual("unlock", classify_talk_page("主界面设置").page)
        task = make_probe_task(仅探测模式=False)
        task.operate_click = Mock()
        task._read = Mock(return_value=(self.frame, [], "获得好感度特效！ 主界面设置"))
        self.assertEqual("failed", task._advance_stage(1))
        task.operate_click.assert_not_called()

    def test_message_about_skipping_dance_is_chat_not_a_skip_control(self):
        text = "GLU TALK 今日的剩余相遇2 未读消息1 我从来没跳过舞。 发送消息 你还会跳舞？"
        self.assertEqual("list", classify_talk_page(text).page)
        task = make_probe_task(仅探测模式=False)
        task.operate_click = Mock()
        boxes = [
            label("我从来没跳过舞。", 0.724, 0.317, 1920, 1080),
            label("你还会跳舞？", 0.66, 0.75, 1920, 1080),
            label("发送消息", 0.56, 0.70, 1920, 1080),
        ]
        task._read = Mock(
            side_effect=[
                (self.frame, boxes, text),
                (self.frame, [], "GLU TALK 今日的剩余相遇2 未读消息0 EP4.对话结束"),
            ]
        )
        self.assertEqual("ended", task._advance_stage(2))
        task.operate_click.assert_called_once()
        self.assertEqual("格鲁TALKreply", task.operate_click.call_args.kwargs["name"])
        self.assertIsNone(dialogue_decision("跳过", boxes[:1], 1920, 1080))

    def test_reopens_after_each_of_three_conversations(self):
        task = make_probe_task(仅探测模式=False)

        def advance(_stage):
            task._click_total += 3
            return "ended"

        task._advance_stage = Mock(side_effect=advance)
        task._restart_from_home = Mock(return_value=True)
        task._wait_for_next_stage = Mock(side_effect=[True, True, False])
        self.assertTrue(task._scan())
        self.assertEqual(3, task.info["完成对话数"])
        self.assertEqual(3, task._restart_from_home.call_count)
        self.assertEqual(3, task._wait_for_next_stage.call_count)

    def test_initial_old_end_marker_does_not_use_up_one_of_three_conversations(self):
        task = make_probe_task(仅探测模式=False)

        def advance(stage):
            if stage > 1:
                task._click_total += 2
            return "ended"

        task._advance_stage = Mock(side_effect=advance)
        task._restart_from_home = Mock(return_value=True)
        task._wait_for_next_stage = Mock(side_effect=[True, True, True, False])
        self.assertTrue(task._scan())
        self.assertEqual(4, task._advance_stage.call_count)
        self.assertEqual(3, task.info["完成对话数"])

    def test_idle_page_is_reopened_before_deciding_there_is_nothing_to_do(self):
        task = make_probe_task(仅探测模式=False)
        task._advance_stage = Mock(return_value="idle")
        task._restart_from_home = Mock(return_value=True)
        task._wait_for_next_stage = Mock(return_value=False)
        self.assertTrue(task._scan())
        task._restart_from_home.assert_called_once()
        self.assertEqual(0, task.info["完成对话数"])

    def test_delayed_new_message_is_seen_after_reopening(self):
        task = make_probe_task(仅探测模式=False)
        task._last_episode = 3
        task._read = Mock(
            side_effect=[
                (self.frame, [], "GLU TALK 今日的剩余相遇2 未读消息0 EP3.对话结束"),
                (self.frame, [], "GLU TALK 今日的剩余相遇2 未读消息0 EP3.对话结束"),
                (self.frame, [], "GLU TALK 今日的剩余相遇2 未读消息1"),
            ]
        )
        with patch("src.tasks.GlutalkTask.monotonic", side_effect=[0, 0.1, 0.6]):
            self.assertTrue(task._wait_for_next_stage())
        self.assertEqual(3, task._read.call_count)

    def test_remaining_quota_alone_is_not_a_new_conversation(self):
        task = make_probe_task(仅探测模式=False)
        self.assertFalse(
            task._has_new_stage(
                "GLU TALK 今日的剩余相遇2 未读消息0 EP3.对话结束",
                [],
                1920,
                1080,
                3,
            )
        )
        boxes = character_list(1920, 1080)
        self.assertTrue(
            task._has_new_stage(
                "GLU TALK 今日的剩余相遇2 未读消息0 EP3.对话结束",
                boxes,
                1920,
                1080,
                3,
            )
        )

    def test_another_character_can_finish_the_same_numbered_episode(self):
        task = make_probe_task(仅探测模式=False)
        task._last_episode = 3
        task.operate_click = Mock()
        task._read = Mock(
            side_effect=[
                (
                    self.frame,
                    [
                        label("你好", 0.66, 0.75, 1920, 1080),
                        label("发送消息", 0.56, 0.70, 1920, 1080),
                    ],
                    "GLU TALK 今日的剩余相遇1 未读消息1 你好",
                ),
                (self.frame, [], "GLU TALK 今日的剩余相遇1 未读消息0 EP3.对话结束"),
            ]
        )
        self.assertEqual("ended", task._advance_stage(2))

    def test_click_limit_is_not_reported_as_success_and_run_returns_home(self):
        task = make_probe_task(仅探测模式=False, 每段最多点击次数=1)
        task.operate_click = Mock()
        task._read = Mock(
            return_value=(
                self.frame,
                [label("你好", 0.66, 0.75, 1920, 1080), label("发送消息", 0.56, 0.70, 1920, 1080)],
                "GLU TALK 今日的剩余相遇2 未读消息1 你好",
            )
        )
        task._safe_return_home = Mock(return_value=True)
        task.log_completion = Mock()
        self.assertFalse(task.run())
        self.assertIn("未完成", task.info["状态"])
        self.assertIn("点击上限", task.info["状态"])
        task._safe_return_home.assert_called_once()
        task.log_completion.assert_not_called()

    def test_end_after_last_allowed_click_is_still_confirmed_without_another_click(self):
        task = make_probe_task(仅探测模式=False, 每段最多点击次数=1)
        task.operate_click = Mock()
        task._read = Mock(
            side_effect=[
                (
                    self.frame,
                    [
                        label("你好", 0.66, 0.75, 1920, 1080),
                        label("发送消息", 0.56, 0.70, 1920, 1080),
                    ],
                    "GLU TALK 今日的剩余相遇2 未读消息1 你好",
                ),
                (self.frame, [], "GLU TALK 今日的剩余相遇2 未读消息0 EP4.对话结束"),
            ]
        )
        self.assertEqual("ended", task._advance_stage(2))
        task.operate_click.assert_called_once()
