# Contributing

We welcome contributions to Strands Robots!

## Getting Started

1. Fork the repository
2. Clone your fork
3. Create a feature branch
4. Make your changes
5. Submit a pull request

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/robots
cd robots
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

## Code Style

We use `black` for formatting and `flake8` for linting:

```bash
black strands_robots tests
flake8 strands_robots tests
```

## Pull Request Guidelines

- Keep PRs focused and small
- Add tests for new features
- Update documentation
- Follow existing code style

## Reporting Issues

Use [GitHub Issues](https://github.com/strands-labs/robots/issues) for bug reports and feature requests.
