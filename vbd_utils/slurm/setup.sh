#!/usr/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
pip install -r ${SCRIPT_DIR}/requirements.txt
pip uninstall opencv-python -y
pip uninstall opencv-python-headless -y
pip install opencv-python-headless

