#!/bin/bash

set -ex

# sudo apt update
# sudo apt install -y tmate
# tmate -S /tmp/tmate.sock new-session -d
# tmate -S /tmp/tmate.sock wait tmate-ready
# tmate -S /tmp/tmate.sock display -p '#{tmate_ssh}'

pwd
ls
whoami
echo $HOME
ls ~

if [[ -d /dev/shm/venv ]]; then {
    mv /dev/shm/venv /dev/shm/_venv
    rm -Rf /dev/shm/_venv &
} fi

python3 -m pip install virtualenv
virtualenv /dev/shm/venv
source /dev/shm/venv/bin/activate
pip install ".[dev,render]"
pip install torch

# tmate -S /tmp/tmate.sock wait tmate-exit
