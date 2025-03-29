#!/bin/bash
set -e
VENV_NAME="deepset_roberta-base-squad2_env"
cd deepset_roberta-base-squad2
pip install virtualenv

virtualenv ${VENV_NAME}
source ${VENV_NAME}/bin/activate
pip install haystack-ai
pip install transformers
python tester.py