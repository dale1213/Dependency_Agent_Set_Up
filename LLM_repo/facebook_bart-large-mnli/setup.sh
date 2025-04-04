#!/bin/bash
set -e
VENV_NAME="facebook_bart-large-mnli_env"
cd facebook_bart-large-mnli
pip install virtualenv

virtualenv ${VENV_NAME}
source ${VENV_NAME}/bin/activate
pip install transformers==4.30.2
pip install torch==2.0.1
pip install numpy==1.24.3
pip install hf_xet==0.1.0
python tester.py