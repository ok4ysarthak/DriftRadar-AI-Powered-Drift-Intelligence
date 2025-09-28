#!/bin/sh
set -e  # exit if any command fails

cd scripts
python 01_train_baseline_model.py
python 02_drift_injection.py
python 03_drift_detection.py
cd ..
