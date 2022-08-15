"""Utility functions for the Multi-Agent Tracking Environment."""

# pylint: disable=invalid-name

import enum
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


__all__ = [
    'seed_everything',
    'RAD2DEG',
    'DEG2RAD',
    'sin_deg',
    'cos_deg',
    'tan_deg',
    'arcsin_deg',
    'arccos_deg',
    'arctan2_deg',
    'cartesian2polar',
    'polar2cartesian',
    'normalize_angle',
    'Vector2D',
    'Team',
    'Message',
]


def seed_everything(seed: int) -> None:
    """Set the seed for global random number generators."""

    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass

    try:
        import tensorflow as tf  # pylint: disable=import-outside-toplevel
    except ImportError:
        pass
    else:
        tf.random.set_seed(seed)


RAD2DEG = 180.0 / np.pi
r"""Coefficient that converts the radian number to the equivalent number in degrees.
The actual value is :math:`\frac{180}{\pi}`.
"""

DEG2RAD = np.pi / 180.0
r"""Coefficient that converts the number degrees number to the radian equivalent.
The actual value is :math:`\frac{\pi}{180}`.
"""


def sin_deg(x):
    r"""Trigonometric sine **in degrees**, element-wise.

    .. math::
        \sin_{\text{deg}} ( x ) = \sin \left( \frac{\pi}{180} x \right)
    """

    return np.sin(np.deg2rad(x))


def cos_deg(x):
    r"""Trigonometric cosine **in degrees**, element-wise.

    .. math::
        \cos_{\text{deg}} ( x ) = \cos \left( \frac{\pi}{180} x \right)
    """

    return np.cos(np.deg2rad(x))


def tan_deg(x):
    r"""Trigonometric tangent **in degrees**, element-wise.

    .. math::
        \tan_{\text{deg}} ( x ) = \tan \left( \frac{\pi}{180} x \right)
    """

    return np.tan(np.deg2rad(x))


def arcsin_deg(x):
    r"""Trigonometric inverse sine **in degrees**, element-wise.

    .. math::
        \arcsin_{\text{deg}} ( x ) = \frac{180}{\pi} \arcsin ( x )
    """

    return np.rad2deg(np.arcsin(x))


def arccos_deg(x):
    r"""Trigonometric inverse cosine **in degrees**, element-wise.

    .. math::
        \arccos_{\text{deg}} ( x ) = \frac{180}{\pi} \arcsin ( x )
    """

    return np.rad2deg(np.arccos(x))


def arctan2_deg(y, x):
    r"""Element-wise arc tangent of y/x **in degrees**.

    .. math::
        \operatorname{arctan2}_{\text{deg}} ( y, x ) = \frac{180}{\pi} \arctan \left( \frac{y}{x} \right)
    """

    return np.rad2deg(np.arctan2(y, x))


def cartesian2polar(x, y):
    r"""Convert cartesian coordinates to polar coordinates **in degrees**, element-wise.

    .. math::
        \operatorname{cartesian2polar} ( x, y ) = \left( \sqrt{x^2 + y^2}, \operatorname{arctan2}_{\text{deg}} ( y, x ) \right)
    """  # pylint: disable=line-too-long

    return np.array([np.hypot(x, y), arctan2_deg(y, x)])


def polar2cartesian(rho, phi):
    r"""Convert polar coordinates to cartesian coordinates **in degrees**, element-wise.

    .. math::
        \operatorname{polar2cartesian} ( \rho, \phi ) = \left( \rho \cos_{\text{deg}} ( \phi ), \rho \sin_{\text{deg}} ( \phi ) \right)
    """  # pylint: disable=line-too-long

    phi_rad = np.deg2rad(phi)
    return rho * np.array([np.cos(phi_rad), np.sin(phi_rad)])


def normalize_angle(angle):
    """Normalize a angle in degree to :math:`[-180, +180)`."""

    return (angle + 180.0) % 360.0 - 180.0


class Vector2D:  # pylint: disable=missing-function-docstring
    """2D Vector."""

    def __init__(self, vector=None, norm=None, angle=None, origin=None):
        self.origin = origin
        self._vector = None
        self._angle = None
        self._norm = None
        if vector is not None and norm is None and angle is None:
            self.vector = np.asarray(vector, dtype=np.float64)
        elif vector is None and norm is not None and angle is not None:
            self.angle = angle
            self.norm = norm
        else:
            raise ValueError

    @property
    def vector(self):
        if self._vector is None:
            self._vector = polar2cartesian(self._norm, self._angle)
        return self._vector

    @vector.setter
    def vector(self, value):
        self._vector = np.asarray(value, dtype=np.float64)
        self._norm = None
        self._angle = None

    @property
    def x(self):
        return self.vector[0]

    @property
    def y(self):
        return self.vector[-1]

    @property
    def endpoint(self):
        return self.origin + self.vector

    @endpoint.setter
    def endpoint(self, value):
        endpoint = np.asarray(value, dtype=np.float64)
        self.vector = endpoint - self.origin

    @property
    def angle(self):
        if self._angle is None:
            self._angle = arctan2_deg(self._vector[-1], self._vector[0])
        return self._angle

    @angle.setter
    def angle(self, value):
        self._angle = normalize_angle(float(value))
        self._vector = None

    @property
    def norm(self):
        if self._norm is None:
            self._norm = np.linalg.norm(self._vector)
        return self._norm

    @norm.setter
    def norm(self, value):
        angle = self.angle
        self._norm = abs(float(value))
        self._vector = None
        if value < 0.0:
            self.angle = angle + 180.0

    def copy(self):
        return Vector2D(vector=self.vector.copy(), origin=self.origin)

    def __eq__(self, other):
        assert isinstance(other, Vector2D)

        return self.angle == other.angle

    def __ne__(self, other):
        return not self == other

    def __imul__(self, other):
        self.norm = self.norm * other

    def __add__(self, other):
        assert isinstance(other, Vector2D)

        return Vector2D(vector=self.vector + other.vector, origin=self.origin)

    def __sub__(self, other):
        assert isinstance(other, Vector2D)

        return Vector2D(vector=self.vector - other.vector, origin=self.origin)

    def __mul__(self, other):
        return Vector2D(norm=self.norm * other, angle=self.angle, origin=self.origin)

    def __rmul__(self, other):
        return self * other

    def __div__(self, other):
        return Vector2D(norm=self.norm / other, angle=self.angle, origin=self.origin)

    def __pos__(self):
        return self

    def __neg__(self):
        return Vector2D(vector=-self.vector, origin=self.origin)

    def __array__(self):
        return self.vector.copy()


class Team(enum.Enum):
    """Enumeration of teams."""

    CAMERA = 0
    TARGET = 1


@dataclass
class Message:
    """Message class for communication between agents in the same team."""

    sender: int
    """Agent index of the sender."""

    recipient: Optional[int]
    """Agent index of the recipient, leave None for broadcasting."""

    content: Any
    """Message content."""

    team: Team
    """String to indicate the team of agents."""

    broadcasting: bool = False
    """Whether or not to broadcast to all teammates."""

    def __contains__(self, name):
        return name in self.content

    def __getitem__(self, name):
        return self.content[name]

    def __setitem__(self, name, value):
        self.content[name] = value


class SpatialHashmap(defaultdict):  # pylint: disable=missing-class-docstring
    def __init__(self, step):
        super().__init__(set)

        self.step = step

    def hash_key(self, name):  # pylint: disable=missing-function-docstring
        return (int(name[0] / self.step), int(name[1] / self.step))
