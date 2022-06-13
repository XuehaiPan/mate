Observations
------------

The observation of agent is a 1D array consists with 5 parts. And the joint observation of the whole team is a 2D array, which is the stack version of all agents' observations in the team.

.. note::

    All entries in the observations are stored in ``float64``, even if the actual values are integers. The observation spaces are continuous for both teams.


------

.. _Camera Observations:

Camera Observations
"""""""""""""""""""

A camera observation :math:`\boldsymbol{o}_c` consists of 5 parts:

.. tikz:: Camera Observation
    :include: ../figures/camera-observation.tikz

1. The preserved part :math:`\boldsymbol{s}^{\text{prsv}}` contains the some auxiliary information. And all values are constants during the whole episode.
    ``observation[0]`` (:math:`N_{\mathcal{C}}`)
        Number of cameras in the environment. This is a constant after ``env.__init__()``.
    ``observation[1]`` (:math:`N_{\mathcal{T}}`)
        Number of targets in the environment. This is a constant after ``env.__init__()``.
    ``observation[2]`` (:math:`N_{\mathcal{O}}`)
        Number of obstacles in the environment. This is a constant after ``env.__init__()``.
    ``observation[3]`` (:math:`c`)
        The index of the current camera in the team (**in zero-based numbering**). This is a constant after ``env.reset()``.
    ``observation[4:12]`` (:math:`x^{(0)}_w, y^{(0)}_w, \dots, x^{(N_{\mathcal{W}} - 1)}_w, y^{(N_{\mathcal{W}} - 1)}_w`)
        Locations of warehouses. They are constants.
    ``observation[12]`` (:math:`r_w`)
        Radius of warehouses. This is a constant.

2. The **private** state of current camera (see :ref:`Camera States`).
    ``observation[13:22]`` (:math:`\boldsymbol{s}_c^{\text{pvt}}`)
        The **private** state of the current camera (i.e. the :math:`c`-th camera, in zero-based numbering).

3. Masked **public** states of targets with additional flags (see :ref:`Target States`).
    A flag function defined as follows:

    .. math::

        \operatorname{flag}^{(\text{C2T})} (c, t) = \mathbb{I} \left[ \text{the $t$-th target is track by the $c$-th camera} \right] \in \{ 0, 1 \},

    where :math:`c` and :math:`t` is the corresponding indices of the camera and the target in their team (**in zero-based numbering**).

    A target is tracked by a camera when one of the following conditions is true:

        1. The target is in the camera's sector shaped field of view **and** not obscured by any obstacle.
        2. If the target is in the camera's sector shaped field of view but obscured by one or more obstacles, the target can be perceived with a probability. The probability is equal to the *transmittance coefficient* of obstacles defined by the configuration file.

    - If the current camera tracks the :math:`t`-th target (i.e. :math:`\operatorname{flag}^{(\text{C2T})} (c, t) = 1`):

        ``observation[22 + 5 * t:22 + 5 * t + 4] (0 <= t < N_t)`` (:math:`\boldsymbol{s}_t^{\text{pub}}`)
            The **public** state of the :math:`t`-th target.
        ``observation[22 + 5 * t + 4] (0 <= t < N_t)`` (:math:`\operatorname{flag}^{(\text{C2T})} (c, t)`)
            One.

    - Otherwise (i.e. :math:`\operatorname{flag}^{(\text{C2T})} (c, t) = 0`):

        ``observation[22 + 5 * t:22 + 5 * t + 4] (0 <= t < N_t)``
            Filled with zeros.
        ``observation[22 + 5 * t + 4] (0 <= t < N_t)`` (:math:`\operatorname{flag}^{(\text{C2T})} (c, t)`)
            Zero.

4. Masked states of obstacles with additional flags (see :ref:`Obstacle States`).
    A flag function defined as follows:

    .. math::

        \begin{split}
            \operatorname{flag}^{(\text{C2O})} (c, o) & = \mathbb{I} \left[ \text{the $o$-th obstacle can be sensed by the $c$-th camera} \right] \\
            & = \begin{cases}
                1, & {\left\| \vec{\boldsymbol{x}}_c^{(c)} - \vec{\boldsymbol{x}}_o^{(o)} \right\|}_2 \le R_{s,c,\max}^{(c)} + r_o^{(o)}, \\
                0, & \text{otherwise}.
            \end{cases}
        \end{split}

    where :math:`c` and :math:`o` is the corresponding indices of the camera and the obstacle (**in zero-based numbering**) and :math:`\vec{\boldsymbol{x}} = (x, y)`. This flag function will be consistent during the whole episode.

    - If the current camera can sense the :math:`o`-th obstacle (i.e. :math:`\operatorname{flag}^{(\text{C2O})} (c, o) = 1`):

        ``observation[22 + 5 * N_t + 4 * o:22 + 5 * N_t + 4 * o + 3] (0 <= o < N_o)`` (:math:`\boldsymbol{s}_o`)
            The state of the :math:`o`-th obstacle.
        ``observation[22 + 5 * N_t + 4 * o + 3] (0 <= o < N_o)`` (:math:`\operatorname{flag}^{(\text{C2O})} (c, o)`)
            One.

    - Otherwise (i.e. :math:`\operatorname{flag}^{(\text{C2O})} (c, o) = 0`):

        ``observation[22 + 5 * N_t + 4 * o:22 + 5 * N_t + 4 * o + 3] (0 <= o < N_o)``
            Filled with zeros.
        ``observation[22 + 5 * N_t + 4 * o + 3] (0 <= o < N_o)`` (:math:`\operatorname{flag}^{(\text{C2O})} (c, o)`)
            Zero.

5. Masked public states of teammates (including itself for convenience) with additional flags (see :ref:`Camera States`).
    A flag function defined as follows:

    .. math::

        \begin{split}
            \operatorname{flag}^{(\text{C2C})} (c_1, c_2) & = \mathbb{I} \left[ \text{the $c_2$-th camera is perceived by the $c_1$-th camera} \right] \in \{ 0, 1 \},
        \end{split}

    where :math:`c` and :math:`c'` is the corresponding indices of the cameras (**in zero-based numbering**).
    The logic of the flag function :math:`\operatorname{flag}^{(\text{C2C})}` is same as :math:`\operatorname{flag}^{(\text{C2T})}`, i.e., within the camera's sector shaped field of view and not occlude by any obstacle.

    - If the current camera perceives the :math:`d`-th camera (i.e. :math:`\operatorname{flag}^{(\text{C2C})} (c, d) = 1`):

        ``observation[22 + 5 * N_t + 4 * N_o + 7 * d:22 + 5 * N_t + 4 * N_o + 7 * d + 6] (0 <= d < N_c)`` (:math:`\boldsymbol{s}_d^{\text{pub}}`)
            The **public** state of the :math:`s`-th target.
        ``observation[22 + 5 * N_t + 4 * N_o + 7 * d + 6] (0 <= d < N_c)`` (:math:`\operatorname{flag}^{(\text{C2C})} (c, d)`)
            One.

    - Otherwise (i.e. :math:`\operatorname{flag}^{(\text{T2T})} (t, s) = 0`):

        ``observation[22 + 5 * N_t + 4 * N_o + 7 * d:22 + 5 * N_t + 4 * N_o + 7 * d + 6] (0 <= d < N_c)``
            Filled with zeros.
        ``observation[22 + 5 * N_t + 4 * N_o + 7 * d + 6] (0 <= d < N_c)`` (:math:`\operatorname{flag}^{(\text{C2C})} (c, d)`)
            Zero.

.. tikz::
    :include: ../figures/camera-observation-schematic.tikz


.. raw:: html

    <table align="center" style="margin-top: 15px; margin-bottom: 15px; table-layout: fixed; width: 100%;">
        <tr valign="middle">
            <td align="right" style="width:20%;">[Camera (in purple)]</td>
            <td style="width:1%;"></td>
            <td align="left" style="width:70%;">the camera's sector shaped field of view (in green) and the bind spot (in cyan)</td>
        </tr>
        <tr valign="middle">
            <td align="right" style="width:20%;">[Target]</td>
            <td style="width:1%;"></td>
            <td align="left" style="width:70%;">tracked (in yellow), untracked (in red) and might be tracked with the probability of transmittance (in orange)</td>
        </tr>
        <tr valign="middle">
            <td align="right" style="width:20%;">[Obstacle]</td>
            <td style="width:1%;"></td>
            <td align="left" style="width:70%;">sensed (in light gray) and not sensed (in darker gray)</td>
        </tr>
        <tr valign="middle">
            <td align="right" style="width:20%;">[Other Camera]</td>
            <td style="width:1%;"></td>
            <td align="left" style="width:70%;">(not shown in the figure) the other camera can be sensed when in the camera's sector shaped field of view (same as the target)</td>
        </tr>
    </table>

.. note::

    For clarity, the targets are drawn as circles here. The actual radiuses of targets are :math:`0`.

The camera observation can be formulated as:

.. math::

    \boldsymbol{o}_c = \mathbb{C} \left( \left[ \boldsymbol{s}^{\text{psrv}}, s_c^{{\text{pvt}}}, \mathbb{F} \left( \begin{bmatrix} \boldsymbol{s}_{t_1}^{\text{pub}} & 1 \\ \vdots & \vdots \\ \boldsymbol{s}_{t_{N_{\mathcal{T}}}}^{\text{pub}} & 1 \end{bmatrix} \otimes \begin{bmatrix} \text{mask} \\ \vdots \\ \text{mask} \end{bmatrix} \right), \mathbb{F} \left( \begin{bmatrix} \boldsymbol{s}_{o_1} & 1 \\ \vdots & \vdots \\ \boldsymbol{s}_{o_{N_{\mathcal{O}}}} & 1 \end{bmatrix} \otimes \begin{bmatrix} \text{mask} \\ \vdots \\ \text{mask} \end{bmatrix} \right), \mathbb{F} \left( \begin{bmatrix} \boldsymbol{s}_{c_1}^{\text{pub}} & 1 \\ \vdots & \vdots \\ \boldsymbol{s}_{c_{N_{\mathcal{C}}}}^{\text{pub}} & 1 \end{bmatrix} \otimes \begin{bmatrix} \text{mask} \\ \vdots \\ \text{mask} \end{bmatrix} \right) \right] \right),

where :math:`\mathbb{C}` means concatenation, :math:`\mathbb{F}` is the flatten operation, and :math:`\otimes` means element-wise multiplication.
For the mask in each row, it's a binary variable that indicates whether the entity is observable by the current camera.

The shape of a single camera observation is:

.. math::

    \begin{split}
        \text{shape} & = (D^{\text{psrv}} + D_c^{\text{pvt}} + (D_t^{\text{pub}} + 1) \times N_{\mathcal{T}} + (D_o + 1) \times N_{\mathcal{O}} + (D_c^{\text{pub}} + 1) \times N_{\mathcal{C}},) \\
                     & = (13 + 9 + (4 + 1) \times N_{\mathcal{T}} + (3 + 1) \times N_{\mathcal{O}} + (6 + 1) \times N_{\mathcal{C}},) \\
                     & = (22 + 5 N_{\mathcal{T}} + 4 N_{\mathcal{O}} + 7 N_{\mathcal{C}},).
    \end{split}

------

The joint version of camera observation is a 2D array, a stack of observations from all cameras.
The shape of the joint observation is:

.. math::

    \text{joint camera observation shape} = (N_{\mathcal{C}}, 22 + 5 N_{\mathcal{T}} + 4 N_{\mathcal{O}} + 7 N_{\mathcal{C}}).

.. tikz:: Joint Camera Observation
    :include: ../figures/camera-observation-joint.tikz


------

.. _Target Observations:

Target Observations
"""""""""""""""""""

A target observation :math:`\boldsymbol{o}_t` consists of 5 parts:

.. tikz:: Target Observation
    :include: ../figures/target-observation.tikz

1. The preserved part :math:`\boldsymbol{s}^{\text{prsv}}` contains the some auxiliary information. And all values are constants during the whole episode.
    ``observation[0]`` (:math:`N_{\mathcal{C}}`)
        Number of cameras in the environment. This is a constant after ``env.__init__()``.
    ``observation[1]`` (:math:`N_{\mathcal{T}}`)
        Number of targets in the environment. This is a constant after ``env.__init__()``.
    ``observation[2]`` (:math:`N_{\mathcal{O}}`)
        Number of targets in the environment. This is a constant after ``env.__init__()``.
    ``observation[3]`` (:math:`t`)
        The index of the current target in the team (**in zero-based numbering**). This is a constant after ``env.reset()``.
    ``observation[4:12]`` (:math:`x^{(0)}_w, y^{(0)}_w, \dots, x^{(N_{\mathcal{W}} - 1)}_w, y^{(N_{\mathcal{W}} - 1)}_w`)
        Locations of warehouses. They are constants.
    ``observation[12]`` (:math:`r_w`)
        Radius of warehouses. This is a constant.

2. The **private** state of current target (see :ref:`Target States`).
    ``observation[13:27]`` (:math:`\boldsymbol{s}_t^{\text{pvt}}`)
        The **private** state of the current target (i.e. the :math:`t`-th target, in zero-based numbering).

3. Masked **public** states of cameras with additional flags (see :ref:`Camera States`).
    A flag function defined as follows:

    .. math::

        \begin{split}
            \operatorname{flag}^{(\text{T2C})} (t, c) & = \mathbb{I} \left[ \text{the $c$-th camera is within the $t$-th target's sight range} \right] \\
            & = \begin{cases}
                1, & {\left\| \vec{\boldsymbol{x}}_t^{(t)} - \vec{\boldsymbol{x}}_c^{(c)} \right\|}_2 \le R_{s,t}^{(t)} + r_c^{(c)}, \\
                0, & \text{otherwise}.
            \end{cases}
        \end{split}

    where :math:`c` and :math:`t` is the corresponding indices of the camera and the target in their team (**in zero-based numbering**) and :math:`\vec{\boldsymbol{x}} = (x, y)`. This flag function ignores any occlusion by obstacles.

    - If the current target perceives the :math:`c`-th camera (i.e. :math:`\operatorname{flag}^{(\text{T2C})} (t, c) = 1`):

        ``observation[27 + 7 * c:27 + 7 * c + 6] (0 <= c < N_c)`` (:math:`\boldsymbol{s}_c^{\text{pub}}`)
            The **public** state of the :math:`c`-th camera.
        ``observation[27 + 7 * c + 6] (0 <= c < N_c)`` (:math:`\operatorname{flag}^{(\text{T2C})} (t, c)`)
            One.

    - Otherwise (i.e. :math:`\operatorname{flag}^{(\text{T2C})} (t, c) = 0`):

        ``observation[27 + 7 * c:27 + 7 * c + 6] (0 <= c < N_c)``
            Filled with zeros.
        ``observation[27 + 7 * c + 6] (0 <= c < N_c)`` (:math:`\operatorname{flag}^{(\text{T2C})} (t, c)`)
            Zero.

4. Masked states of obstacles with additional flags (see :ref:`Obstacle States`).
    A flag function defined as follows:

    .. math::

        \begin{split}
            \operatorname{flag}^{(\text{T2O})} (t, o) & = \mathbb{I} \left[ \text{the $o$-th obstacle is within the $t$-th target's sight range} \right] \\
             & = \begin{cases}
             1, & {\left\| \vec{\boldsymbol{x}}_t^{(t)} - \vec{\boldsymbol{x}}_o^{(o)} \right\|}_2 \le R_{s,t}^{(t)} + r_o^{(o)}, \\
             0, & \text{otherwise}.
             \end{cases}
        \end{split}

    where :math:`t` and :math:`o` is the corresponding indices of the camera and the obstacle (**in zero-based numbering**) and :math:`\vec{\boldsymbol{x}} = (x, y)`. This flag function ignores any occlusion by other obstacles.

    - If the current target senses the :math:`o`-th obstacle (i.e. :math:`\operatorname{flag}^{(\text{T2O})} (t, o) = 1`):

        ``observation[27 + 7 * N_c + 4 * o:27 + 7 * N_c + 4 * o + 3] (0 <= o < N_o)`` (:math:`\boldsymbol{s}_o`)
            The state of the :math:`o`-th obstacle.
        ``observation[27 + 7 * N_c + 4 * o + 3] (0 <= o < N_o)`` (:math:`\operatorname{flag}^{(\text{T2O})} (t, o)`)
            One.

    - Otherwise (i.e. :math:`\operatorname{flag}^{(\text{T2O})} (t, o) = 0`):

        ``observation[27 + 7 * N_c + 4 * o:27 + 7 * N_c + 4 * o + 3] (0 <= o < N_o)``
            Filled with zeros.
        ``observation[27 + 7 * N_c + 4 * o + 3] (0 <= o < N_o)`` (:math:`\operatorname{flag}^{(\text{T2O})} (t, o)`)
            Zero.

5. Masked public states of teammates (including itself for convenience) with additional flags (see :ref:`Target States`).
    A flag function defined as follows:

    .. math::

        \begin{split}
            \operatorname{flag}^{(\text{T2T})} (t_1, t_2) & = \mathbb{I} \left[ \text{the $t_2$-th target is perceived by the $t_1$-th target} \right] \\
             & = \begin{cases}
             1, & {\left\| \vec{\boldsymbol{x}}_t^{(t_1)} - \vec{\boldsymbol{x}}_t^{(t_2)} \right\|}_2 \le R_{s,t}^{(t_1)}, \\
             0, & \text{otherwise}.
             \end{cases}
        \end{split}

    where :math:`t_1` and :math:`t_2` is the corresponding indices of the targets (**in zero-based numbering**) and :math:`\vec{\boldsymbol{x}} = (x, y)`. This flag function ignores any occlusion by obstacles.

    - If the current target senses the :math:`s`-th target (i.e. :math:`\operatorname{flag}^{(\text{T2T})} (t, s) = 1`):

        ``observation[27 + 7 * N_c + 4 * N_o + 5 * s:27 + 7 * N_c + 4 * N_o + 5 * s + 4] (0 <= s < N_t)`` (:math:`\boldsymbol{s}_s^{\text{pub}}`)
            The **public** state of the :math:`s`-th target.
        ``observation[27 + 7 * N_c + 4 * N_o + 5 * s + 4] (0 <= s < N_t)`` (:math:`\operatorname{flag}^{(\text{T2T})} (t, s)`)
            One.

    - Otherwise (i.e. :math:`\operatorname{flag}^{(\text{T2T})} (t, s) = 0`):

        ``observation[27 + 7 * N_c + 4 * N_o + 5 * s:27 + 7 * N_c + 4 * N_o + 5 * s + 4] (0 <= s < N_t)``
            Filled with zeros.
        ``observation[27 + 7 * N_c + 4 * N_o + 5 * s + 4] (0 <= s < N_t)`` (:math:`\operatorname{flag}^{(\text{T2T})} (t, s)`)
            Zero.

.. tikz::
    :include: ../figures/target-observation-schematic.tikz

.. raw:: html

    <table align="center" style="margin-top: 15px; margin-bottom: 15px; table-layout: fixed; width: 100%;">
        <tr valign="middle">
            <td align="right" style="width:25%;">[Target (in purple)]</td>
            <td style="width:1%;"></td>
            <td align="left" style="width:60%;">the target's field of view (within dotted circle)</td>
        </tr>
        <tr valign="middle">
            <td align="right" style="width:25%;">[Camera]</td>
            <td style="width:1%;"></td>
            <td align="left" style="width:60%;">perceived (in yellow), not perceived (in red)</td>
        </tr>
        <tr valign="middle">
            <td align="right" style="width:25%;">[Obstacle]</td>
            <td style="width:1%;"></td>
            <td align="left" style="width:60%;">sensed (in light gray) and not sensed (in darker gray)</td>
        </tr>
        <tr valign="middle">
            <td align="right" style="width:25%;">[Other Target]</td>
            <td style="width:1%;"></td>
            <td align="left" style="width:60%;">perceived (in yellow), not perceived (in red)</td>
        </tr>
    </table>

.. note::

    For clarity, the targets are drawn as circles here.
    The actual radiuses of targets are :math:`0`.

The target observation can be formulated as:

.. math::

    \boldsymbol{o}_t = \mathbb{C} \left( \left[ \boldsymbol{s}^{\text{psrv}}, s_t^{{\text{pvt}}}, \mathbb{F} \left( \begin{bmatrix} \boldsymbol{s}_{c_1}^{\text{pub}} & 1 \\ \vdots & \vdots \\ \boldsymbol{s}_{c_{N_{\mathcal{C}}}}^{\text{pub}} & 1 \end{bmatrix} \otimes \begin{bmatrix} \text{mask} \\ \vdots \\ \text{mask} \end{bmatrix} \right), \mathbb{F} \left( \begin{bmatrix} \boldsymbol{s}_{o_1} & 1 \\ \vdots & \vdots \\ \boldsymbol{s}_{o_{N_{\mathcal{O}}}} & 1 \end{bmatrix} \otimes \begin{bmatrix} \text{mask} \\ \vdots \\ \text{mask} \end{bmatrix} \right), \mathbb{F} \left( \begin{bmatrix} \boldsymbol{s}_{t_1}^{\text{pub}} & 1 \\ \vdots & \vdots \\ \boldsymbol{s}_{t_{N_{\mathcal{T}}}}^{\text{pub}} & 1 \end{bmatrix} \otimes \begin{bmatrix} \text{mask} \\ \vdots \\ \text{mask} \end{bmatrix} \right) \right] \right),

where :math:`\mathbb{C}` means concatenation, :math:`\mathbb{F}` is the flatten operation, and :math:`\otimes` means element-wise multiplication.
For the mask in each row, it's a binary variable that indicates whether the entity is observable by the current target.

The shape of a single target observation is:

.. math::

    \begin{split}
        \text{shape} & = (D^{\text{psrv}} + D_t^{\text{pvt}} + (D_c^{\text{pub}} + 1) \times N_{\mathcal{C}} + (D_o + 1) \times N_{\mathcal{O}} + (D_t^{\text{pub}} + 1) \times N_{\mathcal{T}},) \\
                     & = (13 + 14 + (6 + 1) \times N_{\mathcal{C}} + (3 + 1) \times N_{\mathcal{O}} + (4 + 1) \times N_{\mathcal{T}},) \\
                     & = (27 + 7 N_{\mathcal{C}} + 4 N_{\mathcal{O}} + 5 N_{\mathcal{T}},).
    \end{split}

------

The joint version of target observation is a 2D array, a stack of observations from all targets.
The shape of the joint observation is:

.. math::

    \text{joint target observation shape} = (N_{\mathcal{T}}, 27 + 7 N_{\mathcal{C}} + 4 N_{\mathcal{O}} + 5 N_{\mathcal{T}}).

.. tikz:: Joint Target Observation
    :include: ../figures/target-observation-joint.tikz


------

Related Resources
    - :doc:`/environment/states`
    - :doc:`/environment/actions`
    - :doc:`/environment/rewards`
    - :doc:`/wrappers`
