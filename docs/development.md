# Development

Hopefully, you have landed here because you would like to help out with the development of Beanis. Whether through adding new features, fixing bugs, or extending documentation, your help is really appreciated! Please read this page carefully. If you have any questions, open an issue on [GitHub](https://github.com/andreim14/beanis/issues).

Also, please read the [Code of Conduct](code-of-conduct.md).

## Setting up the development environment

We assume you are familiar with the general forking and pull request workflow for submitting to open-source projects. If not, don't worry, there are plenty of good guides available. Maybe check out [this one](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow).

All the dependencies and build configs are set in the `pyproject.toml` file. There are two main dependency sections there:

- dependencies: for the dependencies required to run Beanis
- test: for the dependencies required to run tests

To install all required dependencies, including test dependencies, in a virtual environment, run the following command in the root directory of the Beanis project:

```shell
pip install -e .[test]
```

### Redis connection

To run tests and use Beanis in general, you will need an accessible Redis database. All tests assume that the database is hosted locally on port `6379` and do not require authentication.

You can run Redis locally using Docker:

```shell
docker run -d -p 6379:6379 redis:latest
```

Or install Redis natively:
- **macOS**: `brew install redis && brew services start redis`
- **Linux**: `sudo apt-get install redis-server && sudo service redis-server start`
- **Windows**: Use [Redis on Windows](https://redis.io/docs/getting-started/installation/install-redis-on-windows/)

## Testing

Beanis uses [pytest](https://docs.pytest.org) for unit testing. To ensure the stability of Beanis, each added feature must be tested in a separate unit test, even if it looks like other tests are covering it now. This strategy guarantees that:

- All the features will be covered and stay covered.
- There is independence from other features and test cases.

To run the test suite, make sure that you have Redis running and run `pytest`:

```shell
pytest
```

To run tests with coverage:

```shell
pytest --cov=beanis --cov-report=html
```

## Submitting new code

You can submit your changes through a pull request on GitHub. Please take into account the following sections.

### Use pre-commit

To ensure code consistency, Beanis uses Black and Ruff through pre-commit. To set it up, run:

```shell
pre-commit install
```

This will add the pre-commit command to your git's pre-commit hooks and make sure you never forget to run these.

### Single commit

To make the pull request reviewing easier and keep the version tree clean, your pull request should consist of a single commit. It is natural that your branch might contain multiple commits, so you will need to squash these into a single commit. Instructions can be found [here](https://www.internalpointers.com/post/squash-commits-into-one-git) or [here](https://medium.com/@slamflipstrom/a-beginners-guide-to-squashing-commits-with-git-rebase-8185cf6e62ec).

### Add documentation

Please write clear documentation for any new functionality you add. Docstrings will be converted to the API documentation, but more human-friendly documentation might also be needed! See the section below.

### Add tests

All new features and bug fixes must include tests. Tests should:
- Be isolated and independent
- Cover both success and failure cases
- Use descriptive names
- Clean up after themselves (delete test data)

## Working on the documentation

The documentation is located in the `docs/` folder and written in Markdown.

### Documentation structure

- `docs/index.md` - Main documentation landing page
- `docs/getting-started.md` - Getting started guide
- `docs/tutorial/` - Step-by-step tutorials
- `docs/reference/` - API reference, generated from docstrings by mkdocstrings

### Building the documentation

The site uses [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/); the configuration is `mkdocs.yml`.

```shell
pip install -e ".[doc]"
mkdocs serve          # preview at http://127.0.0.1:8000/beanis/
mkdocs gh-deploy      # publish to the gh-pages branch
```

## Project structure

```
beanis/
├── beanis/               # Main package
│   ├── odm/              # ODM core
│   │   ├── documents.py  # Document base class
│   │   ├── fields.py     # Field types
│   │   ├── operators/    # Query operators
│   │   ├── queries/      # Query builders
│   │   └── utils/        # Utilities (encoder, decoder)
│   └── executors/        # Background executors
├── tests/                # Test suite
│   ├── odm/              # ODM tests
│   ├── migrations/       # Migration tests (legacy)
│   └── fastapi/          # FastAPI integration tests
├── docs/                 # Documentation
└── benchmarks/           # Performance benchmarks
```

## Contributing guidelines

### Code style

- Follow PEP 8
- Use type hints for all functions
- Write docstrings for public APIs
- Keep functions small and focused

### Git commit messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- First line should be 50 characters or less
- Reference issues and pull requests liberally

Example:
```
Add custom encoder support for NumPy arrays

- Register encoders/decoders for custom types
- Store type metadata with encoded values
- Auto-register NumPy and PyTorch types

Fixes #123
```

### Pull request process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for your changes
5. Run tests (`pytest`)
6. Run pre-commit checks (`pre-commit run --all-files`)
7. Commit your changes
8. Push to your fork (`git push origin feature/amazing-feature`)
9. Open a pull request

## Performance considerations

When contributing, keep these performance tips in mind:

1. **Use Redis pipelines** - Batch operations when possible
2. **Minimize Redis round trips** - Fetch data in bulk
3. **Profile changes** - Run benchmarks for performance-critical code
4. **Avoid unnecessary validation** - Skip validation on reads when safe
5. **Use msgspec** - It's 2x faster than orjson for serialization

Run benchmarks:
```shell
python benchmarks/benchmark_vs_plain_redis.py
```

## Questions?

If you have questions about contributing, please:

1. Check existing [GitHub issues](https://github.com/andreim14/beanis/issues)
2. Open a new issue with the `question` label
3. Provide context and examples

Thank you for contributing to Beanis! 🎉
