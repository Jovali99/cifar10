#!/usr/bin/env bash
echo "Updating system packages..."
apt update
apt install parallel


echo "Creating Python virtual environment..."
python -m venv venv
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing project dependencies from pyproject.toml..."
pip install .

echo "Installation complete!"
