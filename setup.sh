#!/bin/bash

# Create virtual environment
uv venv --python 3.12

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip with trusted hosts
uv pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade pip

# Install requirements with trusted hosts
uv pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r pyproject.toml

echo "Virtual environment setup complete! 🎉"
echo "To activate the virtual environment, run: source .venv/bin/activate" 