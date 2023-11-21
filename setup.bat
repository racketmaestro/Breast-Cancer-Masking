@echo off

echo Creating virtual environment...
virtualenv breast_cancer_masking_venv

echo Activating virtual environment...
call breast_cancer_masking_venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo Setup complete
