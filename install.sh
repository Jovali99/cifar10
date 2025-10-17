#!/usr/bin/env bash
echo "Updating system packages..."
apt update
apt install parallel


if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing project dependencies from pyproject.toml..."
pip install .

echo "Installation complete!"
