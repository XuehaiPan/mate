# pylint: disable=missing-module-docstring

from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np

from mate import constants as consts
from mate.wrappers.typing import MateEnvironmentType, WrapperMeta, assert_mate_environment


class RenderCommunication(gym.Wrapper, metaclass=WrapperMeta):
    """Draw arrows for intra-team communications in rendering results."""

    def __init__(self, env: MateEnvironmentType, duration: Optional[int] = 20) -> None:
        assert_mate_environment(env)
        assert (
            duration > 0
        ), f'The argument `duration` should be a positive integer. Got duration = {duration}.'

        super().__init__(env)

        self.duration = duration
        self.camera_comm_matrix = np.zeros((env.num_cameras, env.num_cameras), dtype=np.int64)
        self.target_comm_matrix = np.zeros((env.num_targets, env.num_targets), dtype=np.int64)

        self.add_render_callback('communication', self.callback)

    def load_config(self, config: Optional[Union[Dict[str, Any], str]] = None) -> None:
        """Reinitialize the Multi-Agent Tracking Environment from a dictionary mapping or a JSON/YAML file."""

        self.env.load_config(config=config)

        self.__init__(self.env, duration=self.duration)  # pylint: disable=unnecessary-dunder-call

    def reset(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        self.camera_comm_matrix.fill(0)
        self.target_comm_matrix.fill(0)

        return self.env.reset(**kwargs)

    def step(
        self, action: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray], Tuple[float, float], bool, Tuple[List[dict], List[dict]]
    ]:
        self.camera_comm_matrix = np.maximum(self.camera_comm_matrix - 1, 0, dtype=np.int64)
        self.target_comm_matrix = np.maximum(self.target_comm_matrix - 1, 0, dtype=np.int64)
        comm_matrices = (self.camera_comm_matrix, self.target_comm_matrix)

        for matrix, message_buffer in zip(comm_matrices, self.unwrapped.message_buffers):
            for message_packs in message_buffer.values():
                for message in message_packs:
                    matrix[message.sender, message.recipient] = self.duration

        return self.env.step(action)

    # pylint: disable-next=unused-argument
    def callback(self, unwrapped: MateEnvironmentType, mode: str) -> None:
        """Draw communication messages as arrows."""

        import mate.assets.pygletrendering as rendering  # pylint: disable=import-outside-toplevel

        geoms = []

        def draw_arrow(start, end, color, bidirectional=False):
            vec = end - start
            norm = np.linalg.norm(vec)
            if norm == 0.0:
                return
            vec *= max(0.9, (norm - 0.05 * consts.TERRAIN_SIZE) / norm)
            vec1 = np.array([[0.04, 0.015], [-0.015, 0.04]]) @ vec
            vec2 = np.array([[0.04, -0.015], [0.015, 0.04]]) @ vec
            vec1 *= min(1.0, 0.05 * consts.TERRAIN_SIZE / np.linalg.norm(vec1))
            vec2 *= min(1.0, 0.05 * consts.TERRAIN_SIZE / np.linalg.norm(vec2))

            start, end = end - vec, start + vec
            if bidirectional:
                vert = np.array([[0.0, -1.0], [1.0, 0.0]]) @ vec
                vert *= 0.01 * consts.TERRAIN_SIZE / np.linalg.norm(vert)
                start, end = start + vert, end + vert

            lines = [rendering.Line(start, end), rendering.Line(end - vec1, end)]

            if not bidirectional:
                lines.append(rendering.Line(end - vec2, end))

            for line in lines:
                line.add_attr(rendering.LineStyle(0x0F0F))
                line.linewidth.stroke = 2.5
                line.set_color(*color)
                geoms.append(line)

        for sender in range(unwrapped.num_cameras):
            for recipient in range(unwrapped.num_cameras):
                remaining = self.camera_comm_matrix[sender, recipient]
                if remaining > 0:
                    draw_arrow(
                        unwrapped.cameras[sender].location,
                        unwrapped.cameras[recipient].location,
                        color=(0.0, 0.0, 1.0, min(1.0, 1.2 * remaining / self.duration)),
                        bidirectional=(self.camera_comm_matrix[recipient, sender] > 0),
                    )

        for sender in range(unwrapped.num_targets):
            for recipient in range(unwrapped.num_targets):
                remaining = self.target_comm_matrix[sender, recipient]
                if remaining > 0:
                    draw_arrow(
                        unwrapped.targets[sender].location,
                        unwrapped.targets[recipient].location,
                        color=(1.0, 0.0, 0.0, min(1.0, 1.2 * remaining / self.duration)),
                        bidirectional=(self.target_comm_matrix[recipient, sender] > 0),
                    )

        unwrapped.viewer.onetime_geoms[:] = geoms + unwrapped.viewer.onetime_geoms
