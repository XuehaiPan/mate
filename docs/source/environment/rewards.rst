Reward Structure
----------------

Targets are mobile agents that aim to transport cargoes between multiple randomly assigned warehouses while minimizing the exposure to cameras.
The target team will be rewarded when cargo is delivered to its destination.
In the meanwhile, the target team may be punished when covered by the camera network.
On the contrary, the camera team will get the opposite value of the target team reward.

Target Team Reward
""""""""""""""""""

**Notations**: For each cargo weighted with :math:`P`, the price :math:`P = F + B` consists with two parts: the freight :math:`F = \alpha W` and the bounty :math:`B = \beta F = \alpha \beta W`, where :math:`\alpha, \beta > 0` are hyperparameters. The freight :math:`F` is a stable income for the target team while the bounty :math:`B` can be discounted by the opponent team. Since the letter ":math:`t`" is taken for the target agent index, we use :math:`k` for environment time-steps.

Suppose a target agent :math:`t \in \mathcal{T}` receives a cargo :math:`W_t` at time-step :math:`k_s`, and delivers it to the desired warehouse at time-step :math:`k_f`. We define an intuitive reward structure as follows:

1. At time-step :math:`k_s`, set:

.. math::
    F_t = \alpha W_t, \qquad B_t (k_s) = B_t = \beta F_t.

.. note::

    Because a new cargo can only be obtained when the last cargo is successfully transported, the reward at time-step :math:`k_s` is calculated by the last delivery.

2. For time-step :math:`k` that :math:`k_s < k \le k_f`, we define a dense reward as:

.. math::

    \begin{split}
        r_t^{(\text{coverage})} (k) = \begin{cases}
            -1, \quad & \text{$B_t (k - 1) > 0$ and the target $t$ is covered by the camera network $\mathcal{C}$}, \\
            0,  \quad & \text{otherwise}.
        \end{cases}
    \end{split}

The bounty will be updated by :math:`B_t (k) = B_t (k - 1) + r_t^{(\text{coverage})} (k) \ge 0`.
This means that the target will be penalized while covered by the cameras, and its bounty will be discounted at the same time.

3. When the target delivers the cargo, we settle the remaining cargo price:

.. math::

    \begin{split}
        r_t^{(\text{transport})} (k) = \begin{cases}
            0,               \quad & k_s < k < k_f \  & (\text{not delivered}), \\
            F_t + B_t (k_f), \quad & k = k_f       \  & (\text{delivered}).
        \end{cases}
    \end{split}

The target team's reward formulates as:

.. math::

    \begin{split}
        r_{\mathcal{T}} (k) = \sum_{t \in \mathcal{T}} \left[ r_t^{(\text{coverage})} (k) + r_t^{(\text{transport})} (k) \right].
    \end{split}

.. note::

    1. It is fully-cooperative for the agents in the same team, i.e., **team-based rewards only** and there are **no individual rewards**. If one agent does nothing but its teammate gains a reward, it will gain the same amount of reward.
    2. If there are more than one of the targets reach their desired warehouses at the same time step, the rewards will be aggregated (sum) into one scalar as the team reward.

Suppose in time-step :math:`t_s` to :math:`t_f`, the target agent :math:`t \in \mathcal{T}` is tracked by the camera network for :math:`K` steps. We have:

.. math::

    B_t (k_f) = \max(0, B_t - K),

and

.. math::

    \begin{split}
        \sum_{k_s < k \le k_f} \left[ r_t^{(\text{coverage})} (k) + r_t^{(\text{transport})} (k) \right]
        & = \sum_{k_s < k \le k_f} r_t^{(\text{coverage})} (k)  + \sum_{k_s < k \le k_f} r_t^{(\text{transport})} (k) \\
        & = - \left[ B_t - B_t (k_f) \right] + \left[ F_t + B_t (k_f) \right] \\
        & = F_t + B_t - 2 \left[ B_t - B_t (k_f) \right] \\
        & = F_t + B_t - 2 \min (K, B_t) \\
        & = \alpha (1 + \beta) W_t - 2 \min (K, \alpha \beta W_t) \\
        & = \alpha W_t \left[ (1 + \beta) - 2 \min (\frac{K}{\alpha W_t}, \beta) \right],
    \end{split}

therefore

.. math::

    \begin{split}
        \alpha W_t (1 - \beta) \le \sum_{k_s < k \le k_f} \left[ r_t^{(\text{coverage})} (k) + r_t^{(\text{transport})} (k) \right] \le \alpha W_t (1 + \beta).
    \end{split}

.. note::

    The left inequality only holds for the target can transport the cargo to the destination, otherwise the maximum value of the total reward is :math:`- B_t = \min \sum_{k_s < k < k_f} \left[ r_t^{(\text{coverage})} (k) + r_t^{(\text{transport})} (k) \right]` (sum excluding :math:`k = k_f`).

The hyperparameters :math:`(\alpha, \beta)` are chosen as:

.. math::

    \alpha W_{t,\max} = \frac{D}{v_{t,\max}}, \quad \beta = 1,

where :math:`D` is the width of the terrain, which is the minimum distance between the warehouses. The value :math:`v_{\max} \times W_{\max}` is a constant among all target agents (see also :ref:`Target States`).

Camera Team Reward
""""""""""""""""""

As the game is a **two-team zero-sum game**, the camera team's reward is:

.. math::

    r_{\mathcal{C}} (k) = - r_{\mathcal{T}} (k).

Goal Assignment
"""""""""""""""

Initially, all targets hold no cargoes, and the ``goal_bits`` are filled with zeros. Each warehouse has a limited amount of cargo to transport to other warehouses.

Once a target reaches a warehouse:

    1. The target holds no cargoes.
        - The warehouse has remaining cargoes to transport:
            Assign a new goal to the target. Flip the corresponding entry of the target's ``goal_bits`` and update the remainings of the current warehouse.
        - Otherwise:
            Do nothing.

    2. The target holds some cargoes and the warehouse is the desired one.
        First, clear the target's ``goal_bits`` (fill with zeros) and calculate the reward function. Then:

        - The warehouse has remaining cargoes to transport:
            Assign a new goal to the target. Flip the corresponding entry of the target's ``goal_bits`` and update the remainings of the current warehouse.
        - Otherwise:
            Do nothing.

    3. The target holds some cargoes but the warehouse is not the desired one.
        Do nothing.

The episode will be terminated when all cargoes have been transported successfully or reach the maximum episode steps.

.. note::

    The amount of the new cargo assignment is no more than both the count of remaining cargoes and the target's weight limits.


------

Related Resources
    - :doc:`/environment/observations`
