#!/bin/bash

set -ex

# sudo apt update
# sudo apt install -y tmate
# tmate -S /tmp/tmate.sock new-session -d
# tmate -S /tmp/tmate.sock wait tmate-ready
# tmate -S /tmp/tmate.sock display -p '#{tmate_ssh}'

pwd
ls

if [[ -d /tmp/venv ]]; then {
    mv /tmp/venv /tmp/_venv
    rm -Rf /tmp/_venv &
} fi

python3 -m pip install virtualenv
virtualenv /tmp/venv
source /tmp/venv/bin/activate
pip install ".[dev,render]"
pip install torch

# tmate -S /tmp/tmate.sock wait tmate-exit
