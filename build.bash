#!/bin/bash
python3 setup.py bdist_wheel
pip install dist/epsrc_vehicle_model-1.0.0-py3-none-any.whl --force-reinstall

