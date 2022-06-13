States
------

We define the state as the internal attributes of the entities, which may change continuously during the interaction between the agent and the environment.
Considering the partially observable setting, different accessibilities are applied to the entity states.
There are two types of states for cameras and targets, i.e., the **publicly accessible states** and the **privately accessible states**.

The publicly accessible state contains the current status of the entity.
And the privately accessible state contains the ability and/or the goals of the entity in addition.

.. note::

    1. For convenience, we omit the word "accessible". We will use "public state" for the publicly accessible state and "private state" for the privately accessible state in the reset of this document.
    2. All entries in the states are stored in ``float64``, even if the actual values are integers.
    3. In this document, We use zero-based numbering.


------

.. _Camera States:

Camera States
"""""""""""""

The public state of a camera is a 1D array with 5 elements (:math:`D_c^{\text{pub}} = 6`), and the private one has 3 addition values (:math:`D_c^{\text{pvt}} = 9`):

``state[0]`` (:math:`x \in [-1000, +1000]`)
    The :math:`x` coordinate of the camera. This is a constant after ``env.reset()``.
``state[1]`` (:math:`y \in [-1000, +1000]`)
    The :math:`y` coordinate of the camera. This is a constant after ``env.reset()``.
``state[2]`` (:math:`r`)
    The radius of the camera. This is a constant after ``env.reset()``.
``state[3]`` (:math:`R_s \cos \phi`)
    The current sight range :math:`R_s` times the cosine value of current orientation angle :math:`\phi`.
``state[4]`` (:math:`R_s \sin \phi`)
    The current sight range :math:`R_s` times the sine value of current orientation angle :math:`\phi`.
``state[5]`` (:math:`\theta`)
    The current viewing angle of the camera **in degrees**.

------

``state[6]`` (:math:`R_{s,\max}`, **private**)
    The maximum sight range of the camera. This is a constant after ``env.reset()``.
``state[7]`` (:math:`{\Delta \phi}_{\max}`, **private**)
    The maximum rotation step of the camera **in degrees**. This is a constant after ``env.reset()``.
``state[8]`` (:math:`{\Delta \theta}_{\max}`, **private**)
    The maximum zooming step of the camera **in degrees**. This is a constant after ``env.reset()``.

.. tikz:: Camera States
    :include: ../figures/camera-state.tikz

.. tikz::
    :include: ../figures/camera-state-schematic.tikz


------

.. _Target States:

Target States
"""""""""""""

The public state of a target is a 1D array with 4 elements (:math:`D_t^{\text{pub}} = 4`), and the private one has 9 addition values (:math:`D_t^{\text{pub}} = 6 + 2 N_{\mathcal{W}} = 14`):

``state[0]`` (:math:`x \in [-1000, +1000]`)
    The current :math:`x` coordinate of the target.
``state[1]`` (:math:`y \in [-1000, +1000]`)
    The current :math:`y` coordinate of the target.
``state[2]`` (:math:`R_s`)
    The sight range :math:`R_s` of the target. This is a constant after ``env.reset()``.
``state[3]`` (:math:`\mathbb{I} [ \text{is loaded} ] \in \{0, 1\}`)
    A boolean value that indicates whether target is loaded or not.

------

``state[4]`` (:math:`v_{\max}`, **private**)
    The maximum step size of the target. This is a constant after ``env.reset()``.
``state[5]`` (:math:`W_{\max} \in \{1, 2\}`, **private**)
    The capacity of the target. This is a constant after ``env.reset()``.
``state[6:10]`` (``goal_bits``, :math:`(W^{(0)}, \dots, W^{(N_{\mathcal{W}} - 1)})`, **private**)
    An :math:`N_{\mathcal{W}}`-element array to indicate the current cargoes that the target holds.
    Only :math:`1` element is non-zero at most at a time.

    This is very similar to a one-hot vector but not exactly the same.
    The target does not hold any cargo if all elements are zero.

.. math::

    \begin{cases}
        \operatorname{\# \, NON-ZERO} \, (W^{(0)}, \dots, W^{(N_{\mathcal{W}} - 1)}) \le 1, \\
        0 \le W^{(g)} \le W_{\max}, \quad g = 0, \dots, N_{\mathcal{W}} - 1, \\
        v_{\max} \times W_{\max} = \text{constant} (\text{for all targets}).
    \end{cases}

.. note::

    There are two types of targets, one with low speed and large capacity, and the other with high speed and small capacity.
    The high-capacity targets' weight limits are set to :math:`W_{\max} = 2`, and the remainings' are :math:`W_{\max} = 1`.
    The population ratio of the high-capacity targets is controlled by ``high_capacity_target_split`` in the configuration (default: :math:`0.5`).

``state[10:14]`` (``empty_bits``, :math:`(E^{(0)}, \dots, E^{(N_{\mathcal{W}} - 1)})`, **private**)
    An :math:`N_{\mathcal{W}}`-element boolean array to indicate the corresponding warehouse is empty or not.

    The value will update when the target reaches the warehouse, and will remain unchanged when the target is far away.
    That means the target may have false believe that the warehouse is not empty while in fact which is not (see also :ref:`Enhanced Observation`).
    Because there are many other targets interacting with the environment simultaneously.

.. tikz:: Target States
    :include: ../figures/target-state.tikz


------

.. _Obstacle States:

Obstacle States
"""""""""""""""

All attributes of an obstacle are public, i.e., the public state and the private state of an obstacle are identical.
The state of an obstacle is a 1D array with 3 elements  (:math:`D_o = 3`):

``state[0]`` (:math:`x \in [-1000, +1000]`)
    The :math:`x` coordinate of the obstacle. This is a constant after ``env.reset()``.
``state[1]`` (:math:`y \in [-1000, +1000]`)
    The :math:`y` coordinate of the obstacle. This is a constant after ``env.reset()``.
``state[2]`` (:math:`r`)
    The radius of the obstacle. This is a constant after ``env.reset()``.

.. tikz:: Obstacle States (public and private are identical)
    :include: ../figures/obstacle-state.tikz


------

Related Resources
    - :doc:`/environment/observations`
    - :doc:`/environment/actions`
    - :doc:`/wrappers`
