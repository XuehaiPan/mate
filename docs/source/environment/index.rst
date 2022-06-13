The Environment Details
=======================

There are 4 types of entities in a 2D mini-world: :math:`N_{\mathcal{C}}` proactive cameras :math:`\mathcal{C} = {\left\{ c_i \right\}}_{i = 1}^{N_{\mathcal{C}}}`, :math:`N_{\mathcal{T}}` mobile targets :math:`\mathcal{T} = {\left\{ t_i \right\}}_{i = 1}^{N_{\mathcal{T}}}`, :math:`N_{\mathcal{O}}` static obstacles, and :math:`N_{\mathcal{W}} ( = 4)` warehouses saving cargoes.

The reward structure inside MATE naturally courage the emergence of the "Min-Max" nature of a cooperative-competitive multi-agent game.

Camera agents need to maximize their coverage rate collaboratively while minimizing repeated detection on the same target.
In the meanwhile, targets aim to transport cargoes between warehouses and minimize the surveillance from the cameras.
The role of obstacles in the environment is to obstruct the sight of the camera, and at the same time can act as a roadblock to the targets.
The warehouses are distributed at the four corners of the mini-world, and there are several cargoes that need to be delivered by targets to other warehouses.

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    states
    observations
    actions
    rewards
    communication

.. image:: https://user-images.githubusercontent.com/16078332/130274196-9d18563d-6d42-493d-8dac-326b1924d2e3.gif
    :align: center
