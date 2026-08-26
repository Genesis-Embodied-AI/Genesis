#!/bin/bash

set -ex

CACHE_ROOT="${HOME}/.cache"
CACHE_VERSION="$(date -u +%GW%V)"
# Retain previous ISO week on Monday (ISO weekday 1); safe while jobs stay within TIMEOUT_MINUTES (60).
if [[ "$(date -u +%u)" -eq 1 ]]; then
  PREV_VERSION="$(date -u -d '7 days ago' +%GW%V)"
else
  PREV_VERSION=""
fi

mkdir -p \
  "${CACHE_ROOT}/${CACHE_VERSION}/uv" \
  "${CACHE_ROOT}/${CACHE_VERSION}/genesis" \
  "${CACHE_ROOT}/${CACHE_VERSION}/quadrants" \
  "${CACHE_ROOT}/${CACHE_VERSION}/huggingface"

# CACHE_ROOT is shared across concurrent runs of this script.
(
  flock 200
  shopt -s nullglob
  for dir in "${CACHE_ROOT}"/[0-9][0-9][0-9][0-9]W[0-9][0-9]; do
    base="$(basename "${dir}")"
    if [[ "${base}" != "${CACHE_VERSION}" && "${base}" != "${PREV_VERSION}" ]]; then
      echo "Removing expired production cache directory: ${dir}"
      rm -rf "${dir}"
    else
      echo "Keeping production cache directory: ${dir}"
    fi
  done
) 200>"${CACHE_ROOT}/.rotate.lock"

echo "CACHE_VERSION=${CACHE_VERSION}" >> "${GITHUB_ENV}"
