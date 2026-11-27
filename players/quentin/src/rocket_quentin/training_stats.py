"""Compte les issues réellement collectées par SAC pendant l'entraînement."""

from collections import defaultdict
from json import dumps
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


class TrainingStatsCallback(BaseCallback):
    def __init__(self, interval: int, total_steps: int, run_dir: Path):
        super().__init__()
        self.interval = interval
        self.total_steps = total_steps
        self.report_path = run_dir / "training-outcomes.json"
        self.history = []
        self.levels = defaultdict(lambda: {"outcomes": {}, "steps": 0, "landed_steps": 0})

    def _on_training_start(self) -> None:
        self._write_window()

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
            if done:
                level = self.levels[str(info["start_difficulty"])]
                outcome = info["outcome"]
                outcomes = level["outcomes"]
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
                level["steps"] += info["episode_steps"]
                if outcome == "landed":
                    level["landed_steps"] += info["episode_steps"]

        steps = self.num_timesteps
        if steps < self.total_steps and steps % self.interval == 0:
            self._write_window()
        return True

    def _on_training_end(self) -> None:
        self._write_window()

    def _write_window(self) -> None:
        coeff = self.model.log_ent_coef.detach().exp().item()
        self.history.append(
            {
                "steps": self.num_timesteps,
                "entropy_coefficient": coeff,
                "target_entropy": self.model.target_entropy,
                "replay_buffer_size": self.model.replay_buffer.size(),
                "by_start_difficulty": dict(self.levels),
            }
        )
        self.levels.clear()
        self.report_path.write_text(dumps(self.history, indent=2), encoding="utf-8")
