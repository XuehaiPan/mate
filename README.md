# MATE: the Multi-Agent Tracking Environment

<!-- markdownlint-disable html -->

This repo contains the source code of `MATE`, the _**M**ulti-**A**gent **T**racking **E**nvironment_. The full documentation can be found at <https://mate-gym.readthedocs.io>. The full list of implemented agents can be found in section [Implemented Algorithms](#implemented-algorithms). For detailed description, please checkout our paper ([PDF](https://openreview.net/pdf?id=SyoUVEyzJbE), [bibtex](#citation)).

This is an **asymmetric two-team zero-sum stochastic game** with _partial observations_, and each team has multiple agents (multiplayer). Intra-team communications are allowed, but inter-team communications are prohibited. It is **cooperative** among teammates, but it is **competitive** among teams (opponents).

## Installation

```bash
git config --global core.symlinks true  # required on Windows
pip3 install git+https://github.com/XuehaiPan/mate.git#egg=mate
```

**NOTE:** Python 3.7+ is required, and Python versions lower than 3.7 is not supported.

It is highly recommended to create a new isolated virtual environment for `MATE` using [`conda`](https://docs.conda.io/en/latest/miniconda.html):

```bash
git clone https://github.com/XuehaiPan/mate.git && cd mate
conda env create --no-default-packages --file conda-recipes/basic.yaml  # or full-cpu.yaml to install RLlib
conda activate mate
```

## Getting Started

Make the ``MultiAgentTracking`` environment and play!

```python
import mate

# Base environment for MultiAgentTracking
env = mate.make('MultiAgentTracking-v0')
env.seed(0)
done = False
camera_joint_observation, target_joint_observation = env.reset()
while not done:
    camera_joint_action, target_joint_action = env.action_space.sample()  # your agent here (this takes random actions)
    (
        (camera_joint_observation, target_joint_observation),
        (camera_team_reward, target_team_reward),
        done,
        (camera_infos, target_infos)
    ) = env.step((camera_joint_action, target_joint_action))
```

Another example with a built-in single-team wrapper (see also [Built-in Wrappers](#built-in-wrappers)):

```python
import mate

env = mate.make('MultiAgentTracking-v0')
env = mate.MultiTarget(env, camera_agent=mate.GreedyCameraAgent(seed=0))
env.seed(0)
done = False
target_joint_observation = env.reset()
while not done:
    target_joint_action = env.action_space.sample()  # your agent here (this takes random actions)
    target_joint_observation, target_team_reward, done, target_infos = env.step(target_joint_action)
```

<p align="center">
  <img src="https://user-images.githubusercontent.com/16078332/130274196-9d18563d-6d42-493d-8dac-326b1924d2e3.gif" alt="Screencast">
  </br>
  4 Cameras vs. 8 Targets (9 Obstacles)
</p>

### Examples and Demos

[`mate/evaluate.py`](mate/evaluate.py) contains the example evaluation code for the `MultiAgentTracking` environment. Try out the following demos:

```bash
# <MultiAgentTracking<MultiAgentTracking-v0>>(4 cameras, 2 targets, 9 obstacles)
python3 -m mate.evaluate --episodes 1 --config MATE-4v2-9.yaml

# <MultiAgentTracking<MultiAgentTracking-v0>>(4 cameras, 8 targets, 9 obstacles)
python3 -m mate.evaluate --episodes 1 --config MATE-4v8-9.yaml

# <MultiAgentTracking<MultiAgentTracking-v0>>(8 cameras, 8 targets, 9 obstacles)
python3 -m mate.evaluate --episodes 1 --config MATE-8v8-9.yaml

# <MultiAgentTracking<MultiAgentTracking-v0>>(4 cameras, 8 targets, 0 obstacle)
python3 -m mate.evaluate --episodes 1 --config MATE-4v8-0.yaml

# <MultiAgentTracking<MultiAgentTracking-v0>>(0 camera, 8 targets, 32 obstacles)
python3 -m mate.evaluate --episodes 1 --config MATE-Navigation.yaml
```

<table style="margin-top: 15px; margin-bottom: 15px; table-layout: fixed; width: 100%;">
  <tr align="center" valign="middle">
    <td style="width:20%;">4 Cameras </br> vs. 2 Targets </br> (9 obstacles)</td>
    <td style="width:20%;">4 Cameras </br> vs. 8 Targets </br> (9 obstacles)</td>
    <td style="width:20%;">8 Cameras </br> vs. 8 Targets </br> (9 obstacles)</td>
    <td style="width:20%;">4 Cameras </br> vs. 8 Targets </br> (no obstacles)</td>
    <td style="width:20%;">8 Targets Navigation </br> (no cameras)</td>
  </tr>
  <tr align="center" valign="middle">
    <td><img src="https://user-images.githubusercontent.com/16078332/130273683-cd0b8a30-ef8f-4d56-bb8a-ae508d51e0e7.gif"></td>
    <td><img src="https://user-images.githubusercontent.com/16078332/130274196-9d18563d-6d42-493d-8dac-326b1924d2e3.gif"></td>
    <td><img src="https://user-images.githubusercontent.com/16078332/130274314-c04d0be9-3af1-4cb9-a33d-0d99c0eec66b.gif"></td>
    <td><img src="https://user-images.githubusercontent.com/16078332/130274049-7fc02965-f2bd-4d37-9d9f-0c6a8279056a.gif"></td>
    <td><img src="https://user-images.githubusercontent.com/16078332/130274359-52b13fdd-189f-47e9-bc9b-feb924215b3a.gif"></td>
  </tr>
</table>

You can specify the agent classes and arguments by:

```bash
python3 -m mate.evaluate --camera-agent module:class --camera-kwargs <JSON-STRING> --target-agent module:class --target-kwargs <JSON-STRING>
```

You can find the example code for agents in [`examples`](examples). The full list of implemented agents can be found in section [Implemented Algorithms](#implemented-algorithms). For example:

```bash
# Example demos in examples
python3 -m examples.naive

# Use the evaluation script
python3 -m mate.evaluate --episodes 1 --render-communication \
    --camera-agent examples.greedy:GreedyCameraAgent --camera-kwargs '{"memory_period": 20}' \
    --target-agent examples.greedy:GreedyTargetAgent \
    --config MATE-4v8-9.yaml \
    --seed 0
```

<p align="center">
  <img src="https://user-images.githubusercontent.com/16078332/131496988-0044c075-67a9-46cb-99a5-c8d290d0b3e4.gif" alt="Communication">
</p>

You can implement your own custom agents classes to play around. See [Make Your Own Agents](docs/source/getting-started.rst#make-your-own-agents) for more details.

## Environment Configurations

The `MultiAgentTracking` environment accepts a Python dictionary mapping or a configuration file in JSON or YAML format.
If you want to use customized environment configurations, you can copy the default configuration file:

```bash
cp "$(python3 -m mate.assets)"/MATE-4v8-9.yaml MyEnvCfg.yaml
```

Then make some modifications for your own. Use the modified environment by:

```python
env = mate.make('MultiAgentTracking-v0', config='/path/to/your/cfg/file')
```

There are several preset configuration files in [`mate/assets`](mate/assets) directory.

```python
# <MultiAgentTracking<MultiAgentTracking-v0>>(4 camera, 2 targets, 9 obstacles)
env = mate.make('MATE-4v2-9-v0')

# <MultiAgentTracking<MultiAgentTracking-v0>>(4 camera, 8 targets, 9 obstacles)
env = mate.make('MATE-4v8-9-v0')

# <MultiAgentTracking<MultiAgentTracking-v0>>(8 camera, 8 targets, 9 obstacles)
env = mate.make('MATE-8v8-9-v0')

# <MultiAgentTracking<MultiAgentTracking-v0>>(4 camera, 8 targets, 0 obstacles)
env = mate.make('MATE-4v8-0-v0')

# <MultiAgentTracking<MultiAgentTracking-v0>>(0 camera, 8 targets, 32 obstacles)
env = mate.make('MATE-Navigation-v0')
```

You can reinitialize the environment with a new configuration without creating a new instance:

```python
>>> env = mate.make('MultiAgentTracking-v0', wrappers=[mate.MoreTrainingInformation])  # we support wrappers
>>> print(env)
<MoreTrainingInformation<MultiAgentTracking<MultiAgentTracking-v0>>(4 cameras, 8 targets, 9 obstacles)>

>>> env.load_config('MATE-8v8-9.yaml')
>>> print(env)
<MoreTrainingInformation<MultiAgentTracking<MultiAgentTracking-v0>>(8 cameras, 8 targets, 9 obstacles)>
```

Besides, we provide a script [`mate/assets/generator.py`](mate/assets/generator.py) to generate a configuration file with responsible camera placement:

```bash
python3 -m mate.assets.generator --path 24v48.yaml --num-cameras 24 --num-targets 48 --num-obstacles 20
```

See [Environment Customization](docs/source/getting-started.rst#environment-customization) for more details.

## Built-in Wrappers

MATE provides multiple wrappers for different settings. Such as _fully observability_, _discrete action spaces_, _single team multi-agent_, etc. See [Built-in Wrappers](docs/source/wrappers.rst#wrappers) for more details.

<table class="docutils align-default">
  <thead>
    <tr class="row-odd">
      <th class="head" colspan="2">Wrapper</th>
      <th class="head">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr class="row-even">
      <td rowspan="5">observation</td>
      <td><code>EnhancedObservation</code></td>
      <td>
        Enhance the agent’s observation, which sets all observation mask to <code>True</code>.
      </td>
    </tr>
    <tr class="row-odd">
      <td><code>SharedFieldOfView</code></td>
      <td>
        Share field of view among agents in the same team, which applies the <code>or</code> operator over the observation masks. The target agents share the empty status of warehouses.
      </td>
    </tr>
    <tr class="row-even">
      <td><code>MoreTrainingInformation</code></td>
      <td>
        Add more environment and agent information to the <code>info</code> field of <code>step()</code>, enabling full observability of the environment.
      </td>
    </tr>
    <tr class="row-odd">
      <td><code>RescaledObservation</code></td>
      <td>
        Rescale all entity states in the observation to <span class="math">[-1, +1]</span>.
      </td>
    </tr>
    <tr class="row-even">
      <td><code>RelativeCoordinates</code></td>
      <td>
        Convert all locations of other entities in the observation to relative coordinates.
      </td>
    </tr>
    <tr class="row-odd">
      <td rowspan="2">action</td>
      <td><code>DiscreteCamera</code></td>
      <td>
        Allow cameras to use discrete actions.
      </td>
    </tr>
    <tr class="row-even">
      <td><code>DiscreteTarget</code></td>
      <td>
        Allow targets to use discrete actions.
      </td>
    </tr>
    <tr class="row-odd">
      <td rowspan="2">reward</td>
      <td><code>AuxiliaryCameraRewards</code></td>
      <td>
        Add additional auxiliary rewards for each individual camera.
      </td>
    </tr>
    <tr class="row-even">
      <td><code>AuxiliaryTargetRewards</code></td>
      <td>
        Add additional auxiliary rewards for each individual target.
      </td>
    </tr>
    <tr class="row-odd">
      <td rowspan="4">single-team</td>
      <td><code>MultiCamera</code>
      <td rowspan="2">
        Wrap into a single-team multi-agent environment.
      </td>
    </tr>
    <tr class="row-even">
      <td><code>MultiTarget</code></td>
    </tr>
    <tr class="row-odd">
      <td><code>SingleCamera</code></td>
      <td rowspan="2">
        Wrap into a single-team single-agent environment.
      </td>
    </tr>
    <tr class="row-even">
      <td><code>SingleTarget</code></td>
    </tr>
    <tr class="row-odd">
      <td rowspan="5">communication</td>
      <td><code>MessageFilter</code></td>
      <td>
        Filter messages from agents of intra-team communications.
      </td>
    </tr>
    <tr class="row-even">
      <td><code>RandomMessageDropout</code></td>
      <td>
        Randomly drop messages in communication channels.
      </td>
    </tr>
    <tr class="row-odd">
      <td><code>RestrictedCommunicationRange</code></td>
      <td>
        Add a restricted communication range to channels.
      </td>
    </tr>
    <tr class="row-even">
      <td><code>NoCommunication</code></td>
      <td>
        Disable intra-team communications, i.e., filter out all messages.
      </td>
    </tr>
    <tr class="row-odd">
      <td><code>ExtraCommunicationDelays</code></td>
      <td>
        Add extra message delays to communication channels.
      </td>
    </tr>
    <tr class="row-even">
      <td>miscellaneous</td>
      <td><code>RepeatedRewardIndividualDone</code></td>
      <td>
        Repeat the <code>reward</code> field and assign individual <code>done</code> field of <code>step()</code>, which is similar to <a href="https://github.com/openai/multiagent-particle-envs">MPE</a>.
      </td>
    </tr>
  </tbody>
</table>

You can create an environment with multiple wrappers at once. For example:

```python
env = mate.make('MultiAgentTracking-v0',
                wrappers=[
                    mate.EnhancedObservation,
                    mate.MoreTrainingInformation,
                    mate.WrapperSpec(mate.DiscreteCamera, levels=5),
                    mate.WrapperSpec(mate.MultiCamera, target_agent=mate.GreedyTargetAgent(seed=0)),
                    mate.RepeatedRewardIndividualDone,
                    mate.WrapperSpec(mate.AuxiliaryCameraRewards,
                                     coefficients={'raw_reward': 1.0,
                                                   'coverage_rate': 1.0,
                                                   'soft_coverage_score': 1.0,
                                                   'baseline': -2.0}),
                ])
```

## Implemented Algorithms

The following algorithms are implemented in [`examples`](examples):

- **Rule-based:**

  1. **Random** (source: [`mate/agents/random.py`](mate/agents/random.py))
  1. **Naive** (source: [`mate/agents/naive.py`](mate/agents/naive.py))
  1. **Greedy** (source: [`mate/agents/greedy.py`](mate/agents/greedy.py))
  1. **Heuristic** (source: [`mate/agents/heuristic.py`](mate/agents/heuristic.py))

- **Multi-Agent Reinforcement Learning Algorithms:**

  1. **IQL** (<https://arxiv.org/abs/1511.08779>)
  1. **QMIX** (<https://arxiv.org/abs/1803.11485>)
  1. **MADDPG** (MA-TD3) (<https://arxiv.org/abs/1706.02275>)
  1. **IPPO** (<https://arxiv.org/abs/2011.09533>)
  1. **MAPPO** (<https://arxiv.org/abs/2103.01955>)

- _Multi-Agent Reinforcement Learning Algorithms_ with **Multi-Agent Communication:**

  1. **TarMAC** (base algorithm: IPPO) (<https://arxiv.org/abs/1810.11187>)
  1. **TarMAC** (base algorithm: MAPPO)
  1. **I2C** (base algorithm: MAPPO) (<https://arxiv.org/abs/2006.06455>)

- **Population Based Adversarial Policy Learning**, available meta-solvers:

  1. Self-Play (SP)
  1. Fictitious Self-Play (FSP) (<https://proceedings.mlr.press/v37/heinrich15.html>)
  1. PSRO-Nash (NE) (<https://arxiv.org/abs/1711.00832>)

**NOTE:** all learning-based algorithms are tested with [Ray 1.12.0](https://github.com/ray-project/ray) on Ubuntu 20.04 LTS.

## Citation

If you find MATE useful, please consider citing:

```bibtex
@inproceedings{pan2022mate,
  title     = {{MATE}: Benchmarking Multi-Agent Reinforcement Learning in Distributed Target Coverage Control},
  author    = {Xuehai Pan and Mickel Liu and Fangwei Zhong and Yaodong Yang and Song-Chun Zhu and Yizhou Wang},
  booktitle = {Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year      = {2022},
  url       = {https://openreview.net/forum?id=SyoUVEyzJbE}
}
```

## License

MIT License
