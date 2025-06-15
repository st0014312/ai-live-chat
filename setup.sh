#!/bin/bash

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip with trusted hosts
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade pip

# Install requirements with trusted hosts
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

echo "Virtual environment setup complete! 🎉"
echo "To activate the virtual environment, run: source venv/bin/activate" 