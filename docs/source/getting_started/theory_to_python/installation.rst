:doc:`← Theory to Python <index>`

==============
 Installation
==============

This guide provides step-by-step installation instructions for **pykal**.

Prerequisites
-------------

**pykal** requires:

* **Python 3.12 or higher**
* **pip** (Python package installer)
* **Linux** (recommended for ROS2 integration; deviate at your peril)


Basic Install
-------------

Install via pip:

.. code-block:: bash

   pip install pykal

We **strongly recommend** using a virtual environment. Using a virtual environment isolates pykal dependencies from your system Python:

.. code-block:: bash

   # Create a new virtual environment
   python3 -m venv pykal-env

   # Activate the virtual environment
   source pykal-env/bin/activate

   # Install pykal
   pip install pykal

   # Verify installation
   pip list | grep pykal

To deactivate the virtual environment later:

.. code-block:: bash

   deactivate
   
:doc:`← Theory to Python <index>`

