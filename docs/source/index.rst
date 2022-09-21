Welcome to the Multi-Agent Tracking Environment's documentation!
================================================================

Welcome to the documentation of ``MATE``, the *Multi-Agent Tracking Environment*.
The source code of the ``MultiAgentTracking`` environment is hosted on GitHub.
You can find it at `mate <https://github.com/XuehaiPan/mate>`_.
For detailed description, please checkout our paper (`PDF <https://openreview.net/pdf?id=SyoUVEyzJbE>`_, :ref:`bibtex <Citation>`).

This is an **asymmetric two-team zero-sum stochastic game** with *partial observations*, and each team has multiple agents (multiplayer).
Intra-team communications are allowed, but inter-team communications are prohibited.
It is **cooperative** among teammates, but it is **competitive** among teams (opponents).


------

Installation
""""""""""""

.. code:: bash

    git config --global core.symlinks true  # required on Windows
    pip3 install git+https://github.com/XuehaiPan/mate.git#egg=mate

.. note::

    Python 3.7+ is required, and Python versions lower than 3.7 is not supported.

It is highly recommended to create a new isolated virtual environment for ``MATE`` using `conda <https://docs.conda.io/en/latest/miniconda.html>`_:

.. code:: bash

    git clone https://github.com/XuehaiPan/mate.git && cd mate
    conda env create --no-default-packages --file conda-recipes/basic.yaml  # or full-cpu.yaml to install RLlib
    conda activate mate

.. image:: https://user-images.githubusercontent.com/16078332/130274196-9d18563d-6d42-493d-8dac-326b1924d2e3.gif
    :align: center


------

.. toctree::
    :maxdepth: 4
    :caption: Contents:

    getting-started
    environment/index
    wrappers
    modules/index


------

.. _Citation:

Citation
========

If you find MATE useful, please consider citing:

.. code-block:: bibtex

    @inproceedings{pan2022mate,
      title     = {{MATE}: Benchmarking Multi-Agent Reinforcement Learning in Distributed Target Coverage Control},
      author    = {Xuehai Pan and Mickel Liu and Fangwei Zhong and Yaodong Yang and Song-Chun Zhu and Yizhou Wang},
      booktitle = {Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
      year      = {2022},
      url       = {https://openreview.net/forum?id=SyoUVEyzJbE}
    }


------

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
