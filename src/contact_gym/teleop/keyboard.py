from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from pynput import keyboard

from .base import Teleop

if TYPE_CHECKING:
    import gymnasium as gym

    from .base import FloatArray


@dataclass(frozen=True)
class KeyBindings:
    """Keyboard manipulator teleop keybindings."""

    # Translation
    x_pos: str = "d"
    x_neg: str = "a"
    y_pos: str = "w"
    y_neg: str = "s"
    z_pos: str = "e"
    z_neg: str = "q"

    # Rotation
    rx_pos: str = "l"
    rx_neg: str = "j"
    ry_pos: str = "i"
    ry_neg: str = "k"
    rz_pos: str = "o"
    rz_neg: str = "u"

    # Gripper
    gripper_open: str = "x"
    gripper_close: str = "z"

    # Brake
    brake: str = "space"

    # Speed scale factor (clipped to [0, 1])
    speed_up: str = "+"
    speed_down: str = "-"


class Keyboard(Teleop):
    """Keyboard teleoperation task-space (with optional gripper) manipulation environments.

    See `contact_gym.controllers` for task-space control environment wrappers. Auxiliary
    methods `get_status_str` and `print_controls` can be used to get the current teleop
    direction/gripper positon/speed and print control keybindings, respectively.

    Args:
        env: Controlled environment.
        pos_scale: Scale factor for delta positions. Can be used to limit the max linear
            speed. Default value is 1.
        rot_scale: Scale factor for delta rotations. Can be used to limit the max angular
            speed. Default value is 1.
        gripper_step: Gripper position step per `get_action` call. Gripper position is
            updated according to its direction, `gripper_step` and speed factor.
            Default value is 0.2.
        speed_init: Initial value for speed scale factor. Default value is 0.5.
        speed_step: Speed step size for each "speed_up" or "speed_down" key press.
            Default value is 0.1.
        timeout_thr: Thread lock key acquire timeout threshold in seconds. If `get_action`
            fails to acquire the lock within `timeout_thr` seconds, the previous action
            is output unmodified. Default value is 0.1.
        bindings: Teleop keybindings configuration. See :class:`KeyBindings` for
            registered attributes and default bindings. Default value is None.
    """

    def __init__(
        self,
        env: gym.Env,
        pos_scale: float = 1.0,
        rot_scale: float = 1.0,
        gripper_step: float = 0.2,
        speed_init: float = 0.5,
        speed_step: float = 0.1,
        timeout_thr: float = 0.1,
        bindings: KeyBindings | None = None,
    ):
        super().__init__(env)
        assert 0 <= speed_init <= 1, f"Initial speed must be in [0, 1]. Got {speed_init}."
        self.pose_scale = np.array([pos_scale] * 3 + [rot_scale] * 3, dtype=self.dtype)
        self.gripper_step = gripper_step
        self.speed_init = speed_init
        self.speed_step = speed_step
        self.timeout_thr = timeout_thr
        self.bindings = KeyBindings() if bindings is None else bindings

        self._lock = threading.Lock()
        self._direction = np.zeros(6, dtype=self.dtype)
        self._gripper_dir = 0.0
        self._gripper_pos = self.gripper_range[1]
        self._speed = speed_init
        self._action = (
            np.array([0] * 6 + [self._gripper_pos]) if self.has_gripper else np.zeros(6)
        ).astype(dtype=self.dtype)

        # Key -> (axis index, sign)
        self._axis_map = {
            self.bindings.x_pos: (0, 1.0),
            self.bindings.x_neg: (0, -1.0),
            self.bindings.y_pos: (1, 1.0),
            self.bindings.y_neg: (1, -1.0),
            self.bindings.z_pos: (2, 1.0),
            self.bindings.z_neg: (2, -1.0),
            self.bindings.rx_pos: (3, 1.0),
            self.bindings.rx_neg: (3, -1.0),
            self.bindings.ry_pos: (4, 1.0),
            self.bindings.ry_neg: (4, -1.0),
            self.bindings.rz_pos: (5, 1.0),
            self.bindings.rz_neg: (5, -1.0),
        }

        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()

    # region Teleop API

    def get_action(self) -> FloatArray:
        """Get the current action."""
        with self.acquire(timeout=self.timeout_thr) as acquired:
            if not acquired:
                return self._action.copy()
            direction = self._direction.copy()
            speed = self._speed
            if self.has_gripper and self._gripper_dir != 0.0:
                self._gripper_pos = float(
                    np.clip(
                        self._gripper_pos + self._gripper_dir * self.gripper_step * speed,
                        self.gripper_range[0],
                        self.gripper_range[1],
                    )
                )
            gripper_pos = self._gripper_pos

        self._action[:6] = direction * self.pose_scale * speed
        if self.has_gripper:
            self._action[6] = gripper_pos
        return self._action.copy()

    def reset(self) -> None:
        """Reset teleop to a default initial state."""
        with self._lock:
            self._direction[:] = 0.0
            self._gripper_dir = 0.0
            self._gripper_pos = self.gripper_range[1]
            self._speed = self.speed_init
        self._action[:] = 0.0
        if self.has_gripper:
            self._action[6] = self.gripper_range[1]

    def close(self) -> None:
        """Close any reserved resources or additonal threads before shutdown."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    # region Auxiliary

    def get_status_str(self) -> str:
        """Get status string with pose direction, gripper position and speed."""
        with self._lock:
            d = self._direction.copy()
            s = self._speed
            g = self._gripper_pos if self.has_gripper else None
        parts = [
            f"dpos=({d[0]:+.0f},{d[1]:+.0f},{d[2]:+.0f})",
            f"drot=({d[3]:+.0f},{d[4]:+.0f},{d[5]:+.0f})",
        ]
        if g is not None:
            parts.append(f"gripper={g:.2f}")
        parts.append(f"speed={s:.2f}")
        return " | ".join(parts)

    def print_controls(self) -> None:
        """Print keyboard teleop controls according to set keybindings."""
        b = self.bindings
        print("Keyboard teleop controls")
        print(f"\ttranslate +x/-x: {b.x_pos}/{b.x_neg}")
        print(f"\ttranslate +y/-y: {b.y_pos}/{b.y_neg}")
        print(f"\ttranslate +z/-z: {b.z_pos}/{b.z_neg}")
        print(f"\trotate    +x/-x: {b.rx_pos}/{b.rx_neg}")
        print(f"\trotate    +y/-y: {b.ry_pos}/{b.ry_neg}")
        print(f"\trotate    +z/-z: {b.rz_pos}/{b.rz_neg}")
        if self.has_gripper:
            print(f"\tgripper open/close: {b.gripper_open}/{b.gripper_close}")
        print(f"\tbrake: {b.brake}")
        print(f"\tspeed up/down: {b.speed_up}/{b.speed_down}\n")

    # region Helpers

    @contextmanager
    def acquire(self, *, timeout: float = -1):
        try:
            yield self._lock.acquire(timeout=timeout)
        finally:
            self._lock.release()

    @staticmethod
    def _key_to_str(key: keyboard.Key) -> str | None:
        if isinstance(key, keyboard.KeyCode):
            return key.char.lower() if key.char else None
        if key == keyboard.Key.space:
            return "space"
        return None

    def _on_press(self, key: keyboard.Key) -> None:
        k = self._key_to_str(key)
        if k is None:
            return
        b = self.bindings
        with self._lock:
            if k == b.brake:
                self._direction[:] = 0.0
                self._gripper_dir = 0.0
                return

            if k == b.speed_up:
                self._speed = float(np.clip(self._speed + self.speed_step, 0.0, 1.0))
                return
            if k == b.speed_down:
                self._speed = float(np.clip(self._speed - self.speed_step, 0.0, 1.0))
                return

            if self.has_gripper and k == b.gripper_open:
                self._gripper_dir = 1.0
                return
            if self.has_gripper and k == b.gripper_close:
                self._gripper_dir = -1.0
                return

            if k in self._axis_map:
                axis, sign = self._axis_map[k]
                self._direction[axis] = sign
                return
