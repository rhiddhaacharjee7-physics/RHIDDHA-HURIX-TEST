#!/usr/bin/env bash

set +e

python /tests/verify.py

status=$?

mkdir -p /logs/verifier

if [ "$status" -eq 0 ]; then

    echo 1 > /logs/verifier/reward.txt

    exit 0

else

    echo 0 > /logs/verifier/reward.txt

    exit 1

fi
