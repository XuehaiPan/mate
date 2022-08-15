"""Example of MADDPG/MA-TD3 agents for the Multi-Agent Tracking Environment."""

from examples.maddpg import camera, target
from examples.maddpg.camera import MADDPGCameraAgent
from examples.maddpg.target import MADDPGTargetAgent


CameraAgent = MADDPGCameraAgent
TargetAgent = MADDPGTargetAgent
