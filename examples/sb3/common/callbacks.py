from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization

if TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecEnv

# region HParam


class HParamCallback(BaseCallback):
    """Log experiment config to TensorBoard's HPARAMS tab at the start of training.

    Args:
        config: *Raw* config dict as loaded from TOML. Flattened to a single shallow dict
            as required by :class:`HParam`.
        metrics: Metrics that will appear in the `HPARAMS` Tensorboard tab.
    """

    METRICS_DEFAULT: ClassVar[list[str]] = ["rollout/ep_rew_mean", "eval/mean_reward"]

    def __init__(self, config: dict, metrics: list[str] = METRICS_DEFAULT, verbose: int = 0):
        super().__init__(verbose)
        self.config = config
        self.metrics = metrics

    def _on_training_start(self) -> None:
        hparam_dict = self._flatten_config(self.config)
        metrics_dict = {k: 0.0 for k in self.metrics}
        self.logger.record(
            "hparams", HParam(hparam_dict, metrics_dict), exclude=("stdout", "log", "json", "csv")
        )

    def _on_step(self) -> bool:
        return True

    @staticmethod
    def _flatten_config(config: dict, parent_key: str = "", sep: str = "/") -> dict[str, Any]:
        """Flatten a nested dict into a shallow ``{"a/b/c": value}`` mapping.

        Handles dicts, bool, str, float and int. Other types are converted to their string
        representations.
        """
        flat: dict[str, Any] = {}
        for key, value in config.items():
            full_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                flat.update(HParamCallback._flatten_config(value, full_key, sep))
            elif isinstance(value, (bool, str, float, int)) or value is None:
                flat[full_key] = value
            else:
                flat[full_key] = str(value)
        return flat


# region Rollout


class RolloutWithStatsCallback(BaseCallback):
    """Periodic logging of windowed rollout mean/min/max reward, episode length, and
    success rate to TensorBoard.

    Similar to `rollout/ep_rew_mean` and `rollout/ep_len_mean` from SB3's own
    `model.learn()` methods but with additional min/max and success reporting.
    Metrics are computed only over the last `log_interval` episodes, not the full model
    episode buffer.

    Args:
        log_interval: Logging interval in episodes. Must match `model.learn()`'s own
            `log_interval`. Default value is 4.
        verbose: verbosity level: 0 for no output, 1 for info messages, 2 for debug
            messages. Default value is 0.
    """

    def __init__(self, log_interval: int = 4, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self._rewards: list = []
        self._lengths: list = []
        self._successes: list = []

    def _on_step(self) -> bool:
        for info, done in zip(self.locals.get("infos", []), self.locals.get("dones", [])):
            if not done:
                continue
            episode_info = info.get("episode")
            if episode_info is not None:
                self._rewards.append(episode_info["r"])
                self._lengths.append(episode_info["l"])
            is_success = info.get("is_success")
            if is_success is not None:
                self._successes.append(float(is_success))

        n_logs = max(len(self._rewards), len(self._successes))
        if self.log_interval > 0 and n_logs % self.log_interval == 0:
            self._record_stats()
        return True

    def _record_stats(self) -> None:
        if self._rewards:
            self.logger.record("rollout_window/ep_rew_mean", float(np.mean(self._rewards)))
            self.logger.record("rollout_window/ep_rew_min", float(np.min(self._rewards)))
            self.logger.record("rollout_window/ep_rew_max", float(np.max(self._rewards)))
        if self._lengths:
            self.logger.record("rollout_window/ep_len_mean", float(np.mean(self._lengths)))
            self.logger.record("rollout_window/ep_len_min", float(np.min(self._lengths)))
            self.logger.record("rollout_window/ep_len_max", float(np.max(self._lengths)))
        if self._successes:
            self.logger.record("rollout_window/success_rate_mean", float(np.mean(self._successes)))
            self.logger.record("rollout_window/success_rate_min", float(np.min(self._successes)))
            self.logger.record("rollout_window/success_rate_max", float(np.max(self._successes)))

        self._rewards.clear()
        self._lengths.clear()
        self._successes.clear()


# region Eval


class EvalWithStatsCallback(EventCallback):
    """Periodic evaluation on a separate eval env, logging mean/min/max reward, episode
    length, and success rate to TensorBoard, and saving the best model by mean reward.

    Alternative to `stable_baselines3.common.callbacks.EvalCallback` that additionally
    reports min/max, not just the mean. See
    :class:`~stable_baselines3.common.callbacks.EvalCallback` for argument descriptions.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        n_eval_episodes: int = 20,
        eval_freq: int = 10_000,
        deterministic: bool = True,
        best_model_save_path: str | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        assert eval_freq > 0
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.best_model_save_path = (
            None if best_model_save_path is None else Path(best_model_save_path)
        )
        self.best_mean_reward = -np.inf
        self._is_succelog_intervalss_buffer = []

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.best_model_save_path is not None:
            self.best_model_save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        if isinstance(self.training_env, VecNormalize):
            sync_envs_normalization(self.training_env, self.eval_env)
        self._is_success_buffer = []
        episode_rewards, episode_lengths = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
            return_episode_rewards=True,
            callback=self._log_success,
            warn=False,
        )
        episode_rewards = np.asarray(episode_rewards, dtype=np.float64)
        episode_lengths = np.asarray(episode_lengths, dtype=np.float64)

        mean_reward = float(episode_rewards.mean())
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/min_reward", float(episode_rewards.min()))
        self.logger.record("eval/max_reward", float(episode_rewards.max()))
        self.logger.record("eval/mean_ep_length", float(episode_lengths.mean()))
        self.logger.record("eval/min_ep_length", float(episode_lengths.min()))
        self.logger.record("eval/max_ep_length", float(episode_lengths.max()))

        if self._is_success_buffer:
            successes = np.asarray(self._is_success_buffer, dtype=np.float64)
            self.logger.record("eval/success_rate_mean", float(np.mean(successes)))
            self.logger.record("eval/success_rate_min", float(np.min(successes)))
            self.logger.record("eval/success_rate_max", float(np.max(successes)))
        self.logger.dump(self.num_timesteps)

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            if self.best_model_save_path is not None:
                self.model.save(self.best_model_save_path / "model")
                if isinstance(self.training_env, VecNormalize):
                    self.training_env.save(str(self.best_model_save_path / "vecnormalize.pkl"))
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"New best mean reward: {mean_reward:.3f}")
                    print(f"Saving new best model to {self.best_model_save_path}")
        return True

    def _log_success(self, locals_: dict, globals_: dict) -> None:
        info = locals_["info"]
        if locals_["done"]:
            is_success = info.get("is_success")
            if is_success is not None:
                self._is_success_buffer.append(float(is_success))
