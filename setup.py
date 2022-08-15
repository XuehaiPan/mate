"""MATE: The Multi-Agent Tracking Environment."""

import pathlib
import sys

from setuptools import setup


HERE = pathlib.Path(__file__).absolute().parent

sys.path.insert(0, str(HERE / 'mate'))
import version  # pylint: disable=import-error,wrong-import-position


setup(
    name='mate',
    version=version.__version__,
)
