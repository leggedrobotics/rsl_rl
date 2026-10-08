# Contribution

We welcome contributions from the community. Bug fixes and improvements to the core library should target the `main`
branch. For new features, we recommend first opening an issue to discuss the proposed contribution before opening a pull
request to the `extras` branch.

## Code Style

- Follow the [PEP 8](https://peps.python.org/pep-0008/) style guide for code.
- Follow the [Google Style Guide](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for docstrings.
- Use the [ruff](https://github.com/astral-sh/ruff) linter and formatter to maintain code quality.

## Workflow

1. For new features, open an issue to discuss the proposed contribution.
2. Fork the repository and create a branch from `main` for bug fixes, or from `extras` for new features.
3. Implement the contribution. New features should extend the core library rather than modify it. Document the new
   feature in the docs of the `extras` branch and add it to the list of features.
4. Add yourself to the [CONTRIBUTORS.md](https://github.com/leggedrobotics/rsl_rl/blob/main/CONTRIBUTORS.md) file.
5. Run [pre-commit](https://pre-commit.com/) to format and lint code with:

   ```bash
   pre-commit run --all-files
   ```

6. Open a pull request to the `main` branch for bug fixes, or to the `extras` branch for new features.
