#!/usr/bin/env bash
echo -e "Updating system packages..."
apt update
apt install parallel


if [ -d "venv" ]; then
    echo -e "Virtual environment already exists. Skipping creation."
else
    echo -e "Creating Python virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

echo -e "Upgrading pip..."
pip install --upgrade pip

echo -e "Installing project dependencies from pyproject.toml..."
pip install .

echo -e "Installing torch!"
pip install torch==2.7.0+cu126 torchvision==0.22.0+cu126 --index-url https://download.pytorch.org/whl/cu126

echo -e "Installation complete!"
