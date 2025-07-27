@echo off

:: Create virtual environment
python -m venv venv

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Upgrade pip with trusted hosts
python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade pip

:: Install requirements with trusted hosts
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

echo Virtual environment setup complete! 🎉
echo To activate the virtual environment, run: venv\Scripts\activate.bat 