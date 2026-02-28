Contribution
============

We welcome contributions from the community. For new features, we recommend first opening an issue to discuss the 
proposed contribution before opening a pull request.

Code Style
----------
- Follow the `PEP 8 <https://peps.python.org/pep-0008/>`_ style guide for code.
- Follow the `Google Style Guide <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_ for 
  docstrings.
- Use the `ruff <https://github.com/astral-sh/ruff>`_ linter and formatter to maintain code quality.


Workflow
--------
1. For new features, open an issue to discuss the proposed contribution.
2. Fork the repository and implement the contribution.
3. Add yourself to the `CONTRIBUTORS.md <https://github.com/leggedrobotics/rsl_rl/blob/main/CONTRIBUTORS.md>`_ file.
4. Run `pre-commit <https://pre-commit.com/>`_ to format and lint code with:

.. code-block:: bash

   pre-commit run --all-files

5. Open a pull request to the main branch.