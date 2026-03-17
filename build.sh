#!/usr/bin/env bash

clear
set -euo pipefail

echo "Cleaning up processed directory..."
rm -rfv dataset/pcqm4m-v2/processed/*

echo "Building dataset..."
time python src/dataset.py
