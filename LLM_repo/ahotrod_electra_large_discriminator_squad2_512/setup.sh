#!/bin/bash
set -e
VENV_NAME="ahotrod_electra_large_discriminator_squad2_512_env"
cd ahotrod_electra_large_discriminator_squad2_512
pip install virtualenv

virtualenv ${VENV_NAME}
source ${VENV_NAME}/bin/activate
pip install transformers==2.11.0
pip install torch==1.5.0
pip install tensorflow==2.2.0