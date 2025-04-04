#!/bin/bash
pip cache purge

if [ -d ~/.cache/huggingface ]; then
  rm -rf ~/.cache/huggingface/*
fi

find . -type d -name "__pycache__" -exec rm -rf {} +

find . -name "*.pyc" -delete

if [ -d ~/.cache/torch ]; then
  rm -rf ~/.cache/torch/*
fi

if command -v conda >/dev/null 2>&1; then
  conda clean -a -y
fi
"""