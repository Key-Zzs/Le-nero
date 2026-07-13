from pathlib import Path
import sys
import types

import numpy as np
import pyarrow.parquet as pq

# The focused record-loop test does not use simulation, while the dp3 test
# environment intentionally lacks gymnasium. Provide only the annotation types
# imported transitively by lerobot.policies.factory.
if "gymnasium" not in sys.modules:
    gymnasium = types.ModuleType("gymnasium")
    gymnasium.Env = type("Env", (), {})
    gymnasium.vector = types.SimpleNamespace(VectorEnv=type("VectorEnv", (), {}))
    sys.modules["gymnasium"] = gymnasium
if "rerun" not in sys.modules:
    sys.modules["rerun"] = types.ModuleType("rerun")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.teleoperator import Teleoperator


class _Teleop(Teleoperator):
    name = "test_frame_sink"

    @property
    def action_features(self):
        return {"action": float}

    @property
    def feedback_features(self):
        return {}

    @property
    def is_connected(self):
        return True

    def connect(self, calibrate=True):
        return None

    @property
    def is_calibrated(self):
        return True

    def calibrate(self):
        return None

    def configure(self):
        return None

    def get_action(self):
        return {"action": 2.0}

    def send_feedback(self, feedback):
        return None

    def disconnect(self):
        return None


class _Robot:
    robot_type = "test"

    def get_observation(self):
        return {"state": 1.0, "raw_only": np.array([7], dtype=np.int64)}

    def send_action(self, action):
        return action


class _Dataset:
    fps = 1000
    features = {
        "observation.state": {"dtype": "float32", "shape": (1,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (1,), "names": ["action"]},
    }

    def __init__(self):
        self.frames = []

    def add_frame(self, frame):
        self.frames.append(frame)


class _Sink:
    def __init__(self):
        self.calls = []

    def add_frame(self, *, observation, frame):
        self.calls.append((observation, frame))


def _identity(value):
    return value[0] if isinstance(value, tuple) else value


def test_record_loop_calls_frame_sink_exactly_once_only_with_dataset():
    teleop = _Teleop.__new__(_Teleop)
    sink = _Sink()
    dataset = _Dataset()
    events = {"exit_early": False}

    record_loop(
        robot=_Robot(),
        events=events,
        fps=1000,
        teleop_action_processor=_identity,
        robot_action_processor=_identity,
        robot_observation_processor=_identity,
        dataset=dataset,
        teleop=teleop,
        control_time_s=0.0001,
        single_task="test",
        frame_sink=sink,
    )

    assert len(dataset.frames) == 1
    assert len(sink.calls) == 1
    observation, frame = sink.calls[0]
    assert np.array_equal(observation["raw_only"], np.array([7]))
    assert "raw_only" not in frame

    reset_sink = _Sink()
    record_loop(
        robot=_Robot(),
        events={"exit_early": False},
        fps=1000,
        teleop_action_processor=_identity,
        robot_action_processor=_identity,
        robot_observation_processor=_identity,
        dataset=None,
        teleop=teleop,
        control_time_s=0.0001,
        single_task="test",
        frame_sink=reset_sink,
    )
    assert reset_sink.calls == []


def _transaction_dataset(root: Path) -> LeRobotDataset:
    return LeRobotDataset.create(
        repo_id="tests/sealed_episodes",
        fps=30,
        root=root,
        robot_type="test",
        use_videos=False,
        features={
            "observation.state": {"dtype": "float32", "shape": (1,), "names": ["state"]},
            "action": {"dtype": "float32", "shape": (1,), "names": ["action"]},
        },
    )


def _add_episode(dataset: LeRobotDataset, value: float) -> None:
    dataset.add_frame(
        {
            "observation.state": np.array([value], dtype=np.float32),
            "action": np.array([value], dtype=np.float32),
            "task": "test",
        }
    )
    dataset.save_episode()
    dataset.seal_episode_writers()


def test_seal_episode_writers_preserves_prior_parquet_files(tmp_path):
    dataset = _transaction_dataset(tmp_path / "dataset")
    dataset.meta.metadata_buffer_size = 1
    _add_episode(dataset, 1.0)
    _add_episode(dataset, 2.0)
    dataset.finalize()

    data_files = sorted((dataset.root / "data").rglob("*.parquet"))
    episode_files = sorted((dataset.root / "meta" / "episodes").rglob("*.parquet"))
    assert len(data_files) == 2
    assert len(episode_files) == 2
    assert sum(pq.read_metadata(path).num_rows for path in data_files) == 2
    assert sum(pq.read_metadata(path).num_rows for path in episode_files) == 2
