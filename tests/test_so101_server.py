import threading
import unittest
from types import SimpleNamespace

import numpy as np

from examples.so101.host_server_so101 import NORM_TAG, Policy


class _RecordingModel:
    def __init__(self) -> None:
        self.kwargs = None

    def predict_action(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(actions=np.zeros((1, 30, 6), dtype=np.float32))


class So101ServerTest(unittest.TestCase):
    def _policy(self) -> tuple[Policy, _RecordingModel]:
        model = _RecordingModel()
        policy = object.__new__(Policy)
        policy.processor = object()
        policy.model = model
        policy._lock = threading.Lock()
        return policy, model

    def test_policy_uses_so101_checkpoint_kwargs(self) -> None:
        policy, model = self._policy()
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        actions = policy.predict(
            scene_cam=image,
            wrist_cam=image,
            instruction="pick up the lemon",
            state=np.zeros(6, dtype=np.float32),
        )

        self.assertEqual(actions.shape, (30, 6))
        self.assertEqual(model.kwargs["inference_action_mode"], "continuous")
        self.assertEqual(model.kwargs["norm_tag"], NORM_TAG)
        self.assertEqual(len(model.kwargs["images"]), 2)

    def test_policy_rejects_wrong_state_dim(self) -> None:
        policy, _ = self._policy()
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            policy.predict(scene_cam=image, wrist_cam=image, instruction="x",
                           state=np.zeros(8, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
