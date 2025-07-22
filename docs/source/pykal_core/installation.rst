Installation
============

Requirements
------------
- Python 3.8 or newer

Basic install
-------------
From PyPI:

.. code-block:: bash

   pip install pykal

Editable (development) install
------------------------------
If you plan to hack on the code, clone the repo and install in “editable” mode:

.. code-block:: bash

   git clone https://github.com/nehalsinghmangat/pykal.git
   cd pykal
   pip install -e .[dev]

That pulls in:
  - core dependencies (NumPy, SciPy, pandas, Matplotlib)  
  - dev tools (pytest, black, mypy, etc.)  
  - docs tools (Sphinx + extensions)

Conda (optional)
----------------
If you prefer conda:

.. code-block:: bash

   conda create -n pykal python=3.10
   conda activate pykal
   pip install -e .[dev]
