Actions and Dynamics
--------------------

.. note::

    The action spaces are continuous for both teams, but can be wrapped into discrete settings (see :ref:`Discrete Action Spaces`).

Camera Actions
""""""""""""""

The camera supports rotation and zooming operations.
The action of a camera is a pair of float numbers :math:`(\Delta \phi, \Delta \theta)` (**in degrees**).
The input action will be clamped first:

.. math::

    \begin{gathered}
        {\Delta \phi}^* = \max \left( -{\Delta \phi}_{\max}, \min \left( \Delta \phi, {\Delta \phi}_{\max} \right) \right), \\
        {\Delta \theta}^* = \max \left( -{\Delta \theta}_{\max}, \min \left( \Delta \theta, {\Delta \theta}_{\max} \right) \right).
    \end{gathered}

.. tikz:: Camera's Action Space
    :include: ../figures/camera-action.tikz

And the update rules for the camera at next timestep are:

.. math::

    \begin{cases}
        \phi'   = \phi + {\Delta \phi}^* \pm 360^{\circ}, \qquad (\phi' \in \left[ -180^{\circ}, +180^{\circ} \right) ), \\
        \theta' = \max \left( \theta_{\min}, \min \left( \theta + {\Delta \theta}^* , \theta_{\max} \right) \right), \\
        R_s'    = R_{s,\max} \sqrt{\frac{\theta_{\min}}{\theta'}}.
    \end{cases}

The zoom in and out operations keep the area of the camera's field of view, i.e., :math:`\frac{\pi}{360^{\circ}} \cdot \theta' \cdot {R_s'}^2 = \frac{\pi}{360^{\circ}} \cdot \theta_{\min} \cdot R_{s,\max}^2 = \text{constant}`.

.. tikz:: The area of the camera's field of view is preserved.
    :include: ../figures/camera-action-schematic.tikz

The joint version of camera action is a 2D array, a stack of all cameras' action.
The shape of the joint action is :math:`(N_{\mathcal{C}}, 2)`.


------

Target Actions
""""""""""""""

The action for targets is simple, a movement represented in cartesian coordinates.
The input action :math:`\vec{\boldsymbol{v}} = (v_x, v_y)` will be clamped first:

.. math::

    \vec{\boldsymbol{v}}^* = \begin{cases}
        \vec{\boldsymbol{v}},                                                                  & \quad {\left\| \vec{\boldsymbol{v}} \right\|}_2 \le v_{\max}, \\
        \frac{v_{\max}}{{\left\| \vec{\boldsymbol{v}} \right\|}_2} \cdot \vec{\boldsymbol{v}}, & \quad {\left\| \vec{\boldsymbol{v}} \right\|}_2 > v_{\max}. \\
    \end{cases}

.. tikz:: Target's Action Space
    :include: ../figures/target-action.tikz

Then if there are some obstacles in the target's way.
The action will be changed again:

.. tikz:: Collision Handling (keep the tangential component but drop the normal component after collision)
    :include: ../figures/collision-handling.tikz

Finally, the update rules for the target at next timestep are:

.. math::

    \begin{cases}
        x' = \max \left( x_{\min}, \min \left( x + {v_x}^{ ** }, x_{\max} \right) \right), \\
        y' = \max \left( y_{\min}, \min \left( y + {v_y}^{ ** }, y_{\max} \right) \right).
    \end{cases}

The target's coordinates will be clamped into the terrain space.

.. note::

    1. The radiuses of targets are :math:`0`. Targets never collide with each other.
    2. There is a circular barrier at the center of each camera, the collision rule treats this barrier as an obstacle.
    3. In order to reduce the computational cost, the minimum distance between the obstacles is set to :math:`v_{\max}`, which means the target will never collide with more than one obstacle in a single step.
    4. The obstacles never overlap the terrain boundary, and the target location will not be caught inside obstacles.

The joint version of target action is a 2D array, a stack of all targets' action.
The shape of the joint action is :math:`(N_{\mathcal{T}}, 2)`.


------

Related Resources
    - :doc:`/environment/states`
    - :doc:`/wrappers`
