# Contributing to DQN LunarLander

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)

## 📜 Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow:

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different viewpoints and experiences

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/SaintJeane/lunar-lander-dqn.git
   cd lunar-lander-dqn
   ```
3. **Set up the development environment**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

## 🛠️ Development Setup

### Prerequisites

- Python 3.8+
- CUDA (optional, for GPU acceleration)
- Docker (optional)

### Install Development Dependencies

```bash
pip install -r requirements.txt
pip install pytest pytest-cov flake8 black jupyter
```

### Verify Installation

```bash
make test
```

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:

- **Clear title** describing the bug
- **Detailed description** of the problem
- **Steps to reproduce** the issue
- **Expected behavior** vs actual behavior
- **Environment details** (OS, Python version, etc.)
- **Error messages** or screenshots if applicable

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:

- **Clear description** of the proposed feature
- **Use case** explaining why it would be useful
- **Possible implementation** approach (optional)
- **Alternatives considered** (optional)

### Types of Contributions

We welcome various types of contributions:

1. **Algorithm Improvements**
   - Implementing Double DQN, Dueling DQN, etc.
   - Optimizing hyperparameters
   - Adding new exploration strategies

2. **Code Quality**
   - Refactoring for better readability
   - Adding type hints
   - Improving documentation
   - Writing more tests

3. **Infrastructure**
   - CI/CD improvements
   - Docker optimizations
   - Logging and monitoring

4. **Documentation**
   - Fixing typos and errors
   - Adding examples
   - Improving explanations
   - Translating documentation

5. **Performance**
   - Speed optimizations
   - Memory efficiency
   - GPU utilization

## 📝 Coding Standards

### Style Guide

This project follows PEP 8 with some modifications:

- **Line length**: 100 characters (not 79)
- **Quotes**: Single quotes for strings (unless double quotes needed)
- **Imports**: Grouped and sorted (stdlib, third-party, local)

### Code Formatting

We use `black` for code formatting:

```bash
black . --line-length=100
```

### Linting

We use `flake8` for linting:

```bash
flake8 . --max-line-length=100
```

### Type Hints

Please add type hints to all new functions:

```python
def train_agent(episodes: int, learning_rate: float) -> DQNAgent:
    """Train a DQN agent."""
    pass
```

### Documentation

All functions should have docstrings:

```python
def calculate_reward(state: np.ndarray, action: int) -> float:
    """
    Calculate reward for state-action pair.
    
    Args:
        state: Current environment state
        action: Action taken by agent
    
    Returns:
        Reward value as float
    """
    pass
```

## 🧪 Testing

### Running Tests

Run all tests:
```bash
pytest test_dqn.py -v
```

Run with coverage:
```bash
pytest test_dqn.py --cov=. --cov-report=html
```

### Writing Tests

When adding new features, please include tests:

```python
def test_new_feature():
    """Test description."""
    # Setup
    agent = DQNAgent()
    
    # Execute
    result = agent.new_feature()
    
    # Assert
    assert result is not None
```

### Test Structure

Tests should follow the Arrange-Act-Assert pattern:

1. **Arrange**: Set up test data and dependencies
2. **Act**: Execute the code being tested
3. **Assert**: Verify the results

## 🔄 Pull Request Process

### Before Submitting

1. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following coding standards

3. **Add tests** for new functionality

4. **Run tests** to ensure everything passes:
   ```bash
   make test
   ```

5. **Format code**:
   ```bash
   make format
   ```

6. **Commit changes** with clear messages:
   ```bash
   git commit -m "Add feature: description of changes"
   ```

### Commit Message Guidelines

Format: `<type>: <description>`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat: implement Double DQN algorithm
fix: correct Q-value calculation in update step
docs: add training tips to README
test: add unit tests for ReplayBuffer
```

### Submitting the PR

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request** on GitHub with:
   - Clear title describing the change
   - Detailed description of what changed and why
   - Reference to any related issues
   - Screenshots/GIFs if UI changes

3. **Wait for review** and address feedback

### PR Review Process

- Maintainers will review your PR within 1-2 weeks
- You may be asked to make changes
- Once approved, your PR will be merged

## 🎯 Development Workflow

Typical workflow for contributing:

```bash
# 1. Update main branch
git checkout main
git pull upstream main

# 2. Create feature branch
git checkout -b feature/amazing-feature

# 3. Make changes and commit
git add .
git commit -m "feat: add amazing feature"

# 4. Run tests
make test

# 5. Push to your fork
git push origin feature/amazing-feature

# 6. Open Pull Request on GitHub
```

## 📚 Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [DQN Paper](https://www.nature.com/articles/nature14236)
- [OpenAI Spinning Up](https://spinningup.openai.com/)

## ❓ Questions?

If you have questions about contributing:

- Open an issue with the `question` label
- Check existing issues and discussions
- Reach out to maintainer(s)

Thank you for contributing! 🎉