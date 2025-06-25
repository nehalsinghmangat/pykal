.. image:: https://github.com/nehalsinghmangat/pykal/actions/workflows/doctest.yml/badge.svg
    :target: https://github.com/nehalsinghmangat/pykal/actions/workflows/doctest.yml
    :alt: Doctest Status

pykal: Kalman Filtering Framework
=================================

**pykal** is a flexible and extensible Python framework for implementing Kalman filtering and its variants. It supports continuous-time and discrete-time systems, exact or empirical Jacobians, square-root methods, and observability-aware updates.

Overview
--------

The **Kalman Filter (KF)** is a recursive state estimator for linear dynamical systems under Gaussian noise assumptions. It computes the optimal estimate of the hidden system state from noisy measurements, balancing prediction and correction steps using a probabilistic framework. 

For nonlinear systems, extensions like the **Extended Kalman Filter (EKF)** and **Unscented Kalman Filter (UKF)** generalize the filter using linearizations or sigma-point transforms. Additional variants like the **Schmidt-Kalman Filter (SKF)** or **Partial-Update Kalman Filter (PUKF)** adapt to structure, observability, or computational constraints.

Motivation
----------

The core motivation behind `pykal` is to combine:

- ✅ **The expressive flexibility of Python**, allowing users to define system dynamics as ordinary Python functions, neural networks, symbolic expressions, or even numerical integrators;
- ✅ **The safety and structure of statically typed systems**, with runtime validation and type-hint enforcement inspired by C-like robustness;
- ✅ **Research-oriented extensibility**, so that Kalman filter variants (e.g., observability-aware updates, partial updates, square-root filters) can be easily added, tested, and deployed.

This makes `pykal` ideal for:
- Research prototyping in control, robotics, and navigation;
- Educational use in estimation theory;
- Lightweight use in simulation and modeling workflows.

Features
--------

- 🧠 EKF, UKF, SKF, PSKF, PUKF, OPSKF variants
- 🔁 Continuous and discrete-time simulation support
- ⚠️ Observability-aware state updates
- 🔬 Empirical and symbolic Jacobian fallback
- 🧮 Full support for square-root filtering (SRKF, SRIF)
- 🧰 Input validation with rich doctest coverage
- 📈 Integrated simulation of state and measurement trajectories
- 🔌 Plug-and-play system function definitions (`f`, `h`, `Q`, `R`, etc.)

Installation
------------

Install via pip (once published):

.. code-block:: bash

   pip install pykal

Or clone directly for development:

.. code-block:: bash

   git clone https://github.com/nehalsinghmangat/pykal.git
   cd pykal
   pip install -e ".[dev]"

Documentation
-------------

Full documentation is available at: https://pykal.readthedocs.io

License
-------

MIT License

----


