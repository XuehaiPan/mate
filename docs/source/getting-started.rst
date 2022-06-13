Getting Started
===============

This is an **asymmetric two-team zero-sum stochastic game** with *partial observations*, and each team has multiple agents (multiplayer).
Intra-team communications are allowed, but inter-team communications are prohibited.

Make the ``MultiAgentTracking`` environment and play!

.. code:: python

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

Another example with a built-in single-team wrapper (see also :doc:`/wrappers`):

.. code:: python

    import mate

    env = mate.make('MultiAgentTracking-v0')
    env = mate.MultiTarget(env, camera_agent=mate.GreedyCameraAgent(seed=0))
    env.seed(0)
    done = False
    target_joint_observation = env.reset()
    while not done:
        target_joint_action = env.action_space.sample()  # your agent here (this takes random actions)
        target_joint_observation, target_team_reward, done, target_infos = env.step(target_joint_action)


------

Examples and Demos
""""""""""""""""""

:gitcode:`mate/evaluate.py` contains the example evaluation code for the ``MultiAgentTracking`` environment.
Try out the following demos:

.. code:: bash

    # <MultiAgentTracking<MultiAgentTracking-v0>>(4 cameras, 2 targets, 9 obstacles)
    python3 mate.evaluate --episodes 1 --config MATE-4v2-9.yaml

    # <MultiAgentTracking<MultiAgentTracking-v0>>(4 cameras, 8 targets, 9 obstacles)
    python3 mate.evaluate --episodes 1 --config MATE-4v8-9.yaml

    # <MultiAgentTracking<MultiAgentTracking-v0>>(8 cameras, 8 targets, 9 obstacles)
    python3 mate.evaluate --episodes 1 --config MATE-8v8-9.yaml

    # <MultiAgentTracking<MultiAgentTracking-v0>>(4 cameras, 8 targets, 0 obstacle)
    python3 mate.evaluate --episodes 1 --config MATE-4v8-0.yaml

    # <MultiAgentTracking<MultiAgentTracking-v0>>(0 camera, 8 targets, 32 obstacles)
    python3 mate.evaluate --episodes 1 --config MATE-Navigation.yaml

.. raw:: html

    <table style="margin-top: 15px; margin-bottom: 15px; table-layout: fixed; width: 100%;">
        <tr align="center" valign="middle">
            <td style="width:20%;">4 Cameras </br> vs. 2 Targets </br> (9 obstacles)</td>
            <td style="width:20%;">4 Cameras </br> vs. 8 Targets </br> (9 obstacles)</td>
            <td style="width:20%;">8 Cameras </br> vs. 8 Targets </br> (9 obstacles)</td>
            <td style="width:20%;">4 Cameras </br> vs. 8 Targets </br> (9 obstacles)</td>
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

.. code:: bash

    python3 -m mate.evaluate --camera-agent module:class --camera-kwargs <JSON-STRING> --target-agent module:class --target-kwargs <JSON-STRING>

You can find the example code for agents at :gitcode:`examples`.
For example:

.. code:: bash

    # Example demos in examples
    python3 -m examples.naive

    # Use the evaluation script
    python3 -m mate.evaluate --episodes 1 --render-communication \
        --camera-agent examples.greedy:GreedyCameraAgent --camera-kwargs '{"memory_period": 20}' \
        --target-agent examples.greedy:GreedyTargetAgent \
        --config MATE-8v8-9.yaml \
        --seed 0

.. image:: https://user-images.githubusercontent.com/16078332/131496988-0044c075-67a9-46cb-99a5-c8d290d0b3e4.gif
    :align: center

------

Environment Customization
"""""""""""""""""""""""""

The `MultiAgentTracking` environment accepts a Python dictionary mapping or a configuration file in JSON or YAML format.
If you want to use customized environment configurations, you can copy the default configuration file:

.. code:: bash

    cp "$(python3 -m mate.assets)"/MATE-4v8-9.yaml MyEnvCfg.yaml

Then make some modifications for your own.
Use the modified environment by:

.. code:: python

    env = mate.make('MultiAgentTracking-v0', config='/path/to/your/cfg/file')

There are several preset configuration files in ``mate/assets`` directory.

.. code:: python

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

Besides, you can reinitialize the environment with a new configuration without creating a new instance:

.. code:: python

    >>> env = mate.make('MultiAgentTracking-v0', wrappers=[mate.MoreTrainingInformation])  # we support wrappers
    >>> print(env)
    <MoreTrainingInformation<MultiAgentTracking<MultiAgentTracking-v0>>(4 cameras, 8 targets, 9 obstacles)>
    >>> env.load_config('MATE-8v8-9.yaml')
    >>> print(env)
    <MoreTrainingInformation<MultiAgentTracking<MultiAgentTracking-v0>>(8 cameras, 8 targets, 9 obstacles)>

Besides, we provide a script :gitcode:`mate/assets/generator.py` to generate a configuration file with responsible camera placement:

.. code:: bash

    python3 -m mate.assets.generator --path 24v48.yaml --num-cameras 24 --num-targets 48 --num-obstacles 20

Here is an example configuration file in YAML format:

.. code:: yaml

    name: MATE-4C-2T-4O                # the environment name
    max_episode_steps: 10000           # maximum number of episode steps
    num_cargoes_per_target: 8          # average number (>=4) of cargoes per target
    high_capacity_target_split: 0.5    # population ratio of the high-capacity target in the target team (set to 0.5 when not given)
    targets_start_with_cargoes: true   # always assign cargoes to targets at the beginning of an episode (set to true when not given)
    bounty_factor: 1.0                 # ratio of the maximum bounty reward over the freight reward (set to 1.0 when not given)
    shuffle_entities: true             # shuffle entity IDs when reset the environment (set to true when not given)

    camera:                            # *** DELETE THIS ENTRY FOR NO CAMERAS ***
      location:                        # cameras at fixed locations
        - [ 0, 0 ]
      location_random_range:           # random range for cameras' locations on reset()
        - [  500,  800,  500,  800 ]   # [ x_low, x_high, y_low, y_high ]
        - { low: [  500, -800 ], high: [ 800, -500 ] }  # alternative form to specify a range
        - [ -800, -500, -800, -500 ]
        - [ -800, -500,  500,  800 ]

      min_viewing_angle: 30.0          # minimum viewing angle of cameras in degrees
      max_sight_range: 1500.0          # maximum sight range of cameras
      rotation_step: 5.0               # maximum rotation step of cameras in degrees
      zooming_step: 2.5                # maximum zooming step of cameras in degrees
      radius: 40.0                     # radius of the circular barrier (set to 40.0 when not given)

    target:                            # *** THERE MUST BE AT LEAST ONE TARGET IN THE ENVIRONMENT ***
      location_random_range:           # random range for targets' locations on reset()
        - [ -200,  200, -200,  200 ]   # [ x_low, x_high, y_low, y_high ]
        - [ -200,  200, -200,  200 ]
        - [   80,   80,    0,    0 ]   # fixed initial location

      step_size: 20.0                  # maximum step size of targets
      sight_range: 500.0               # sight range of targets

    obstacle:                          # *** DELETE THIS ENTRY FOR NO OBSTACLES ***
      location_random_range:           # random range for obstacles' locations on reset()
        - [  200,  800,  200,  800 ]   # [ x_low, x_high, y_low, y_high ]
        - [  200,  800, -800, -200 ]
        - [ -800, -200, -800, -200 ]
        - [ -800, -200,  200,  800 ]

      radius_random_range: [ 25, 100 ]  # random range for obstacles' radiuses on reset()
      # radius: 75                      # replace the above random range with this for fixed-sized obstacles
      transmittance: 0.1                # transmittance coefficient of obstacles (set to 0.0 when not given)

Here is the same example configuration file in JSON format (the comments should be removed before pasting to a JSON file):

.. code:: javascript

    {
      "name": "MATE-4C-2T-4O",             // the environment name
      "max_episode_steps": 10000,          // maximum number of episode steps
      "num_cargoes_per_target": 8,         // average number (>=4) of cargoes per target
      "high_capacity_target_split": 0.5    // population ratio of the high-capacity target in the target team (set to 0.5 when not given)
      "targets_start_with_cargoes": false  // always assign cargoes to targets at the beginning of an episode (set to false when not given)
      "bounty_factor": 1.0                 // ratio of the maximum bounty reward over the freight reward
      "shuffle_entities": true             // shuffle entity IDs when reset the environment (set to true when not given)

      "camera": {                          // *** DELETE THIS ENTRY FOR NO CAMERAS ***
        "location": [                      // cameras at fixed locations
          [ 0, 0 ]
        ],
        "location_random_range": [         // random range for cameras' locations on reset()
          [  500,  800,  500,  800 ],      // [ x_low, x_high, y_low, y_high ]
          { "low": [  500, -800 ], "high": [ 800, -500 ] },  // alternative form to specify a range
          [ -800, -500, -800, -500 ],
          [ -800, -500,  500,  800 ]
        ],
        "min_viewing_angle": 30.0,         // minimum viewing angle of cameras in degrees
        "max_sight_range": 1500.0,         // maximum sight range of cameras
        "rotation_step": 5.0,              // maximum rotation step of cameras in degrees
        "zooming_step": 2.5                // maximum zooming step of cameras in degrees
        "radius": 40.0                     // radius of the circular barrier (set to 40.0 when not given)
      },

      "target": {                          // *** THERE MUST BE AT LEAST ONE TARGET IN THE ENVIRONMENT ***
        "location_random_range": [         // random range for targets' locations on reset()
          [ -200,  200, -200,  200 ],      // [ x_low, x_high, y_low, y_high ]
          [ -200,  200, -200,  200 ],
          [   80,   80,    0,    0 ]       // fixed initial location
        ],
        "step_size": 20.0,                 // maximum step size of targets
        "sight_range": 500.0               // sight range of targets
      },

      "obstacle": {                        // *** DELETE THIS ENTRY FOR NO OBSTACLES ***
        "location_random_range": [         // random range for obstacles' locations on reset()
          [  200,  800,  200,  800 ],      // [ x_low, x_high, y_low, y_high ]
          [  200,  800, -800, -200 ],
          [ -800, -200, -800, -200 ],
          [ -800, -200,  200,  800 ]
        ],
        "radius_random_range": [ 25, 100 ],  // random range for obstacles' radiuses on reset()
        // "radius": 75,                     // replace the above random range with this for fixed-sized obstacles
        "transmittance": 0.1                 // transmittance coefficient of obstacles (set to 0.0 when not given)
      }
    }

There will be :math:`N_{\mathcal{C}}` (= ``len(cameras["location"])`` + ``len(cameras["location_random_range"])``) cameras in the environment, and the numbers of targets (:math:`N_{\mathcal{T}}`) and obstacles (:math:`N_{\mathcal{O}}`) can be obtained in the same way.
The above example configuration file contains :math:`5` cameras, :math:`3` targets and :math:`4` obstacles.

``shuffle_entities`` indicates whether to shuffle entity IDs when resetting the environment.
We encourage this behavior to make agents not overfit their role selection on the entity ID rather than the entity ability or affordance.
At the beginning of each episode (on ``reset()``), the entities in then the environment will be shuffled.
For example, the camera agent 0 controls the camera at the southeast of the terrain in one episode, and then camera agent 0 may control another camera (e.g. at the northwest) in another episode.

If users set ``shuffle_entities=False``, the order of the entities controlled by agents will strictly correspond to the order in the configuration.
The first half (if split ratio is :math:`0.5`) of the targets will have higher transport capacity but lower movement speed, and the rest targets will have lower transport capacity but higher movement speed (see :ref:`Target States`).

.. note::

    1. You can set ``low == high`` of a random range to always get the same value on ``env.reset()``. (Or use keys without ``_random_range`` suffix.)
    2. All values will be sampled **uniformly** in the random range.
    3. There **MUST** be at least one target in the environment, and no constraints for the number of cameras and obstacles.
    4. ``num_cargoes_per_target`` must be no less than the number of warehouses :math:`N_{\mathcal{W}} \, (= 4)`.
    5. When ``targets_start_with_cargoes`` is true (by default), the environment will always assign cargoes to targets at the beginning of the episode. Otherwise, only targets whose starting location is in the warehouse will be loaded, the others remain empty.
    6. You can set ``high_capacity_target_split = 0.0`` to have only one kind of target in the environment.


------

.. _Make Your Own Agents:

Make Your Own Agents
""""""""""""""""""""

Make your agent classes use inheritance.
You can find the example code at :gitcode:`examples` and :gitcode:`mate/agents`.

.. code:: python

    import copy
    from typing import Iterable, Union, Optional, Any

    import numpy as np

    import mate


    class CameraAgent(mate.CameraAgentBase):
        """Camera agent class."""

        def __init__(self, seed: Optional[int] = None, **kwargs) -> None:
            """Initialize the agent.
            This function will be called only once on initialization.

            Note:
                Agents can obtain the number of teammates and opponents on reset,
                but not here. You are responsible for writing scalable policies and
                code to handle this.
            """

            super().__init__(seed=seed)

            # TODO: Initialize your agent (e.g., build a deep neural network).
            # Put your code here.

        def reset(self, observation: np.ndarray) -> None:
            """Reset the agent.
            This function will be called immediately after env.reset().

            Note:
                observation is a 1D array, not a 2D array with an additional
                dimension for agent indices.
            """

            super().reset(observation)

            # TODO: Do something necessary at the beginning of each episode.
            # Put your code here.

        def act(self, observation: np.ndarray, info: Optional[dict] = None,
                deterministic: Optional[bool] = None) -> Union[int, np.ndarray]:
            """Get the agent action by the observation.
            This function will be called before every env.step().

            Note:
                observation is a 1D array, not a 2D array with an additional
                dimension for agent indices.
            """

            self.state, observation, info, messages = self.check_inputs(observation, info)

            # TODO: Implement your policy here.
            # Put your code here and override the above line.
            return self.action_space.sample()  # this takes random action

        def send_messages(self) -> Iterable[Message]:
            """Prepare messages to communicate with other agents in the same team.
            This function will be called before receive_messages() and act().
            """

            # TODO: Communicate with teammates.
            # Put your code here and override the above lines.
            condition = True  # communicate when necessary
            if not condition:
                return []

            content: Any = f'This is a message from <Camera {self.index}>.'
            to_prev_agent = self.pack_message(recipient=(self.index - 1) % self.num_cameras,
                                              content=content)
            to_next_agent = self.pack_message(recipient=(self.index + 1) % self.num_cameras,
                                              content=content)
            return [to_prev_agent, to_next_agent]

        def receive_messages(self, messages: Tuple[Message, ...]) -> None:
            """Receive messages from other agents in the same team.
            This function will be called after receive_messages() but before act().
            """

            self.last_messages = tuple(messages)

            # TODO: Process received messages and update agent state

        def clone(self) -> 'CameraAgent':
            """Clone an independent copy of the agent."""

            # TODO: Replace this if you want all agents use a shared deep neural network
            clone = copy.deepcopy(self)
            clone.seed(self.np_random.randint(np.iinfo(int).max))
            return clone


    class TargetAgent(mate.TargetAgentBase):
        """Target agent class."""

        def __init__(self, seed: Optional[int] = None, **kwargs) -> None:
            """Initialize the agent.
            This function will be called only once on initialization.

            Note:
                Agents can obtain the number of teammates and opponents on reset,
                but not here. You are responsible for writing scalable policies and
                code to handle this.
            """

            super().__init__(seed=seed)

            # TODO: Initialize your agent (e.g., build a deep neural network).
            # Put your code here.

        def reset(self, observation: np.ndarray) -> None:
            """Reset the agent.
            This function will be called immediately after env.reset().

            Note:
                observation is a 1D array, not a 2D array with an additional
                dimension for agent indices.
            """

            super().reset(observation)

            # TODO: Do something necessary at the beginning of each episode.
            # Put your code here.

        def act(self, observation: np.ndarray, info: Optional[dict] = None,
                deterministic: Optional[bool] = None) -> Union[int, np.ndarray]:
            """Get the agent action by the observation.
            This function will be called before every env.step().

            Note:
                observation is a 1D array, not a 2D array with an additional
                dimension for agent indices.
            """

            self.state, observation, info, messages = self.check_inputs(observation, info)

            # TODO: Implement your policy here.
            # Put your code here and override the above line.
            return self.action_space.sample()  # this takes random action

        def send_messages(self) -> Iterable[Message]:
            """Prepare messages to communicate with other agents in the same team.
            This function will be called before receive_messages() and act().
            """

            # TODO: Communicate with teammates.
            # Put your code here and override the above lines.
            return []

        def receive_messages(self, messages: Tuple[Message, ...]) -> None:
            """Receive messages from other agents in the same team.
            This function will be called after receive_messages() but before act().
            """

            self.last_messages = tuple(messages)

            # TODO: Process received messages and update agent state

.. hint::

    You can preprocess the agent's observation using ``mate.convert_coordinates``, ``mate.rescale_observation`` and/or ``mate.split_observation`` before feeding it into a neural network.

    1. ``mate.convert_coordinates``: Convert all locations of entities in the observation to relative coordinates (exclude the agent itself).
    2. ``mate.rescale_observation``: Rescale all entity states in the observation to :math:`[-1, +1]`.
    3. ``mate.split_observation``: Split the concatenated observations into parts.

    For convenience, you can use:

    .. code:: python

        relative_coordinates = agent.convert_coordinates(observation)
        rescaled_observation = agent.rescale_observation(observation)
        split_observation    = agent.split_observation(observation)

        # split_observation should be placed at last
        all_preprocessing_methods = agent.split_observation(agent.rescale_observation(agent.convert_coordinates(observation)))


------

Related Resources
    - :doc:`/environment/observations`
    - :doc:`/environment/actions`
    - :doc:`/environment/rewards`
    - :doc:`/wrappers`
