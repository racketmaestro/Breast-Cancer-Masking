#!/bin/bash

echo "Creating virtual environment..."
virtualenv breast_cancer_masking_venv

echo "Activating virtual environment..."
source breast_cancer_masking_venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete"
