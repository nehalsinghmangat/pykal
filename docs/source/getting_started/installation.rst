Installation
============

This guide provides step-by-step installation instructions for **pykal** and its optional dependencies.

Prerequisites
-------------

**pykal** requires:

* **Python 3.12 or higher**
* **pip** (Python package installer)
* **Linux** (recommended for ROS2 integration) or macOS (core features only)

Quick Start: Basic Installation
--------------------------------

For basic functionality (e.g. DynamicalSystem), install via pip:

.. code-block:: bash

   pip install pykal

We **strongly recommend** using a virtual environment (see next section).

Recommended: Installation with Virtual Environment
---------------------------------------------------

Using a virtual environment isolates pykal dependencies from your system Python:

.. code-block:: bash

   # Create a new virtual environment
   python3 -m venv pykal-env

   # Activate the virtual environment
   source pykal-env/bin/activate

   # Install pykal
   pip install pykal

   # Verify installation
   python -c "from pykal import DynamicalSystem; print('pykal installed successfully!')"

To deactivate the virtual environment later:

.. code-block:: bash

   deactivate

Optional: ROS2 Installation
----------------------------

To use pykal's ROS2 integration (``ROSNode``, message conversion, Gazebo simulation), you need ROS2 installed.

**Supported Configurations:**

* **Ubuntu 24.04 LTS** → ROS2 Jazzy Jalisco (recommended)
* **Ubuntu 22.04 LTS** → ROS2 Humble Hawksbill

.. note::
   ROS2 is optional. Core pykal functionality (``DynamicalSystem``) works without ROS2. The ``ROSNode`` class and ROS utilities will only fail if actually used without ROS2 installed.

Ubuntu 24.04: ROS2 Jazzy Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Ensure UTF-8 locale
   locale
   sudo apt update && sudo apt install locales
   sudo locale-gen en_US en_US.UTF-8
   sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
   export LANG=en_US.UTF-8

   # Enable required repositories
   sudo apt install software-properties-common
   sudo add-apt-repository universe

   # Add ROS2 GPG key
   sudo apt update && sudo apt install curl -y
   sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
       -o /usr/share/keyrings/ros-archive-keyring.gpg

   # Add ROS2 repository
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
       | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

   # Install ROS2 Jazzy Desktop
   sudo apt update
   sudo apt upgrade
   sudo apt install ros-jazzy-desktop

   # Install development tools
   sudo apt install ros-dev-tools

   # Source ROS2 setup (add this to ~/.bashrc for persistence)
   source /opt/ros/jazzy/setup.bash

   # Install Python ROS2 dependencies
   pip install rclpy

Ubuntu 22.04: ROS2 Humble Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Ensure UTF-8 locale
   locale
   sudo apt update && sudo apt install locales
   sudo locale-gen en_US en_US.UTF-8
   sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
   export LANG=en_US.UTF-8

   # Enable required repositories
   sudo apt install software-properties-common
   sudo add-apt-repository universe

   # Add ROS2 GPG key
   sudo apt update && sudo apt install curl -y
   sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
       -o /usr/share/keyrings/ros-archive-keyring.gpg

   # Add ROS2 repository
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
       | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

   # Install ROS2 Humble Desktop
   sudo apt update
   sudo apt upgrade
   sudo apt install ros-humble-desktop

   # Install development tools
   sudo apt install ros-dev-tools

   # Source ROS2 setup (add this to ~/.bashrc for persistence)
   source /opt/ros/humble/setup.bash

   # Install Python ROS2 dependencies
   pip install rclpy

Make ROS2 Sourcing Permanent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add the ROS2 setup to your ``~/.bashrc`` to automatically source it in new terminals:

.. code-block:: bash

   # For ROS2 Jazzy (Ubuntu 24.04)
   echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc

   # OR for ROS2 Humble (Ubuntu 22.04)
   echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

   # Reload your shell configuration
   source ~/.bashrc

Optional: Gazebo Installation
------------------------------

For robot simulation capabilities, install Gazebo. The version depends on your ROS2 distribution.

Gazebo Harmonic (for ROS2 Jazzy on Ubuntu 24.04)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install Gazebo Harmonic
   sudo apt update
   sudo apt install ros-jazzy-ros-gz

   # Verify installation
   gz sim --version

Gazebo Garden (for ROS2 Humble on Ubuntu 22.04)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install Gazebo Garden
   sudo apt update
   sudo apt install ros-humble-ros-gz

   # Verify installation
   gz sim --version

Install Robot-Specific Packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For TurtleBot3 simulation:

.. code-block:: bash

   # For ROS2 Jazzy
   sudo apt install ros-jazzy-turtlebot3*

   # For ROS2 Humble
   sudo apt install ros-humble-turtlebot3*

   # Set default TurtleBot3 model (add to ~/.bashrc)
   echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc
   source ~/.bashrc

For Crazyflie simulation:

.. code-block:: bash

   # Install Crazyflie ROS2 packages
   # (Instructions vary by ROS2 version - consult Crazyflie documentation)
   sudo apt install ros-${ROS_DISTRO}-crazyflie*

Development Installation
-------------------------

For contributing to pykal or running tests:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/nehalsinghmangat/pykal.git
   cd pykal

   # Create and activate virtual environment
   python3 -m venv .venv
   source .venv/bin/activate

   # Install in editable mode with dev dependencies
   pip install -e ".[dev]"

   # Run tests to verify installation
   pytest

   # Optional: Install pre-commit hooks
   pre-commit install

Documentation Build Setup
--------------------------

To build the documentation locally:

.. code-block:: bash

   # Install with documentation dependencies
   pip install -e ".[docs]"

   # Build documentation
   cd docs
   make html

   # View documentation (open in browser)
   xdg-open build/html/index.html  # Linux
   # or
   open build/html/index.html      # macOS

Verification
------------

Verify your installation with these quick tests:

.. code-block:: bash

   # Test core functionality
   python -c "from pykal import DynamicalSystem; print('✓ Core pykal works')"

   # Test ROS2 integration (only if ROS2 installed)
   python -c "from pykal import ROSNode; print('✓ ROS2 integration works')"

   # Test Gazebo utilities (only if ROS2 installed)
   python -c "from pykal.gazebo import start_gazebo, stop_gazebo; print('✓ Gazebo utilities work')"

   # Test Kalman filter
   python -c "from pykal.estimators import KF; print('✓ Estimators work')"

Troubleshooting
---------------

**ImportError: No module named 'rclpy'**

ROS2 is not installed or not sourced. Either:

1. Install ROS2 (see ROS2 Installation section)
2. Source ROS2 setup: ``source /opt/ros/<distro>/setup.bash``
3. Only use core pykal features (avoid ``ROSNode``)

**Python version too old**

.. code-block:: bash

   # Check Python version
   python3 --version

   # Should show Python 3.12.x or higher
   # If not, install Python 3.12+

**Gazebo fails to launch**

.. code-block:: bash

   # Ensure Gazebo is installed
   gz sim --version

   # Check ROS-Gazebo bridge is installed
   ros2 pkg list | grep ros_gz

   # Verify robot packages
   ros2 pkg list | grep turtlebot3

**Virtual environment issues**

.. code-block:: bash

   # Deactivate current environment
   deactivate

   # Remove old environment
   rm -rf pykal-env

   # Create fresh environment
   python3 -m venv pykal-env
   source pykal-env/bin/activate
   pip install --upgrade pip
   pip install pykal

**Tests fail after development installation**

.. code-block:: bash

   # Ensure all dev dependencies are installed
   pip install -e ".[dev]"

   # Update pytest
   pip install --upgrade pytest pytest-doctestplus

   # Run with verbose output
   pytest -v

Backward Compatibility
----------------------

**pykal** follows semantic versioning. The current version (0.1) is in active development:

* **API stability**: Core APIs (``DynamicalSystem``, ``ROSNode``) are stable
* **ROS2 compatibility**: Tested with Humble (Ubuntu 22.04) and Jazzy (Ubuntu 24.04)
* **Python compatibility**: Requires Python ≥3.12 
* **Breaking changes**: Will be documented in release notes before v1.0

For production use, we recommend:

1. Pin specific versions in your ``requirements.txt``
2. Watch the GitHub repository for release notifications
3. Test new versions in a virtual environment before upgrading

Next Steps
----------

After installation, explore:

* :doc:`../index` - Framework overview and philosophy
* :doc:`algorithms_as_dynamical_systems_latex/algorithms_as_dynamical_systems` - Core concepts
* :doc:`../notebooks/pykal_workflow` - Complete workflow tutorial
* :doc:`../api/index` - API reference

Need Help?
----------

* **Documentation**: https://pykal.readthedocs.io
* **GitHub Issues**: https://github.com/nehalsinghmangat/pykal/issues
* **GitHub Repository**: https://github.com/nehalsinghmangat/pykal
