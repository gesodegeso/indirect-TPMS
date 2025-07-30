# Contributing to Indirect TPMS

Thank you for your interest in contributing to the Indirect TPMS project! This document provides guidelines for contributing.

## How to Contribute

### Reporting Issues

1. Check existing issues to avoid duplicates
2. Use the issue template when available
3. Provide detailed description with:
   - Environment details (Python version, OS, etc.)
   - Steps to reproduce
   - Expected vs actual behavior
   - Code samples if applicable

### Submitting Changes

1. **Fork the repository**
   ```bash
   git clone https://github.com/gesodegeso/indirect-TPMS.git
   cd indirect-TPMS
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the coding standards below
   - Add tests for new functionality
   - Update documentation as needed

4. **Test your changes**
   ```bash
   python -m pytest tests/
   python examples/basic_usage.py
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

6. **Push and create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Coding Standards

### Python Style
- Follow PEP 8 style guide
- Use type hints where possible
- Maximum line length: 88 characters
- Use descriptive variable and function names

### Documentation
- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Update README.md if adding new features

### Testing
- Write unit tests for new functionality
- Maintain test coverage above 80%
- Test edge cases and error conditions

## Development Setup

1. **Clone and install in development mode**
   ```bash
   git clone https://github.com/gesodegeso/indirect-TPMS.git
   cd indirect-TPMS
   pip install -e .[dev]
   ```

2. **Run tests**
   ```bash
   pytest tests/ -v --cov=src
   ```

3. **Code formatting**
   ```bash
   black src/ tests/
   flake8 src/ tests/
   ```

## Types of Contributions

### Priority Areas
- **Algorithm improvements**: Better feature extraction, ML models
- **Performance optimization**: Faster processing, memory efficiency
- **Real vehicle integration**: CAN bus interfaces, hardware support
- **Testing**: More comprehensive test coverage
- **Documentation**: Examples, tutorials, API docs

### New Features
Before implementing major new features:
1. Open an issue to discuss the proposal
2. Wait for maintainer feedback
3. Implement with tests and documentation

## Code Review Process

1. All changes require pull request review
2. Automated tests must pass
3. At least one maintainer approval required
4. Follow-up on review feedback promptly

## Questions?

Feel free to open an issue for questions about contributing!