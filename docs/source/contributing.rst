Contributing to pykal
=====================

We welcome contributions! Whether you're fixing bugs, improving documentation, or proposing new features, your input is valued.

How to Contribute
-----------------

1. **Fork the Repository**

   Start by forking the project on GitHub and cloning your fork locally.

   .. code-block:: bash

      git clone https://github.com/your-username/pykal.git
      cd pykal

2. **Set Up the Development Environment**

   Use the provided `requirements-dev.txt` or install dependencies manually.

   .. code-block:: bash

      python -m venv .venv
      source .venv/bin/activate
      pip install -r requirements-dev.txt

3. **Run Tests**

   All contributions must pass tests and doctests.

   .. code-block:: bash

      pytest
      pytest --doctest-modules

4. **Style Guide**

   - Code must conform to `black` formatting and `flake8` style.
   - Type annotations are required for all public functions and classes.
   - Docstrings should follow NumPy or Google-style conventions.

5. **Write Descriptive Commits**

   Use clear, concise commit messages that explain _why_ a change was made.

6. **Open a Pull Request**

   Submit your changes via a GitHub pull request.
   Be sure to link relevant issues and describe the problem or feature addressed.

7. **Documentation**

   If you add or change public functionality, update the Sphinx `.rst` docs:

   .. code-block:: bash

      make html

   and check the output under `docs/_build/html`.

Contributing Guidelines
-----------------------

- Follow Python best practices (PEP8, PEP484, PEP257).
- Avoid breaking backward compatibility.
- Keep code modular and testable.
- Add unit tests for new functionality.
- Prefer clarity over cleverness.

Questions?
----------

Open an issue or discussion on GitHub — we’re happy to help.

