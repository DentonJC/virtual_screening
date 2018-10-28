#!/usr/bin/env bash

echo "export PYTHONPATH=\$PYTHONPATH:$PWD" >>~/.bashrc
echo "export MKL_THREADING_LAYER=GNU" >>~/.bashrc
