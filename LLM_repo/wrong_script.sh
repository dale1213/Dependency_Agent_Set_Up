#!/bin/bash

# Check if running on a system with apt-get
if command -v apt-get &> /dev/null; then
    # Install pip if not already installed
    which pip3 > /dev/null || (sudo apt-get update && sudo apt-get install -y python3-pip)
else
    # For systems without apt-get, assume pip is installed or notify the user
    echo "Please ensure pip3 is installed on your system."
fi

# Upgrade pip to the latest version
pip3 install --upgrade pip

# Install TensorFlow
pip3 install tensorflow

# Run the Python script
python3 wrong_script.py