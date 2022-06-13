Communication
-------------

MATE implements an explicit intra-team peer-to-peer communication channel for each team.
Agents can send messages to the others in the same team.
Apart from the observation, the agent can also obtain extra information from other agents.

Different from the widely used `Multi-Agent Particle Environment (MPE) <https://github.com/openai/multiagent-particle-envs>`_ that in the all-to-all architecture, we explicitly isolate messages from agent observations, and the agents can communicate with specific peers.
So agents are allowed to share observation, knowledge, and intention with teammates.
This can enable more efficient cooperation between agents, reducing unnecessary exploration in a partially observable environment.

.. code:: python

    # Peer-to-peer
    env.send_messages(mate.Message(sender=0, recipient=1, team=mate.Team.CAMERA,
                                   content='Greetings from camera 0 to 1.'))
    # Broadcasting
    env.send_messages(mate.Message(sender=1, recipient=None, team=mate.Team.TARGET,
                                   content=np.array([0, 0], dtype=np.float64)))

The environment will put the messages to recipients' ``info`` field of ``step()``.
(See also :ref:`Make Your Own Agents` for more information.)

.. tikz:: Multi-round Communication
    :include: ../figures/communication.tikz

.. image:: https://user-images.githubusercontent.com/16078332/131475096-5b283f9e-5b98-41ca-aed1-6d732d3d5d3a.png
    :align: center
    :width: 600

Here is a typical scene of communication demonstrated above.
The cameras transmit the location information of the targets to prevent overlapping.
In the meanwhile, a target is not assigned with a new payload after arriving at a warehouse.
Then it broadcasts the information that this warehouse is empty to its teammates to avoid unnecessary exploration.

Try the following command for a demo:

.. code:: bash

    python3 -m mate.evaluate --episodes 1 --render-communication \
        --camera-agent examples.greedy:GreedyCameraAgent \
        --target-agent examples.greedy:GreedyTargetAgent \
        --config MATE-8v8-9.yaml \
        --seed 0

.. image:: https://user-images.githubusercontent.com/16078332/131496988-0044c075-67a9-46cb-99a5-c8d290d0b3e4.gif
    :align: center


------

Related Resources
    - :doc:`/getting-started`
    - :doc:`/environment/observations`
    - :doc:`/environment/rewards`
    - :doc:`/wrappers`
