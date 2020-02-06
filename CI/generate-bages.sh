#!/usr/bin/env bash
COV=`coverage report | grep TOTAL | awk '{print $4}' |  tr -d %`
echo $COV
anybadge --value="${COV}" --file=coverage.svg coverage
