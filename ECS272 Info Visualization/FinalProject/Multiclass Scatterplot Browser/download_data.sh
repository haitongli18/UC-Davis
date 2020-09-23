#!/bin/bash

function fail {
    printf '%s\n' "$1" >&2  ## Send message to stderr. Exclude >&2 if you don't want it that way.
    exit "${2-1}"  ## Return a code specified by $2 or 1 by default.
}

cd finalproj-django || fail "Can't find finalproj-django directory"
echo "Downloading Person Activity Dataset"
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00196/ConfLongDemo_JSI.txt || fail "Failed to download dataset"
cat ./data/person_activity_headers.txt ConfLongDemo_JSI.txt > ConfLongDemo_JSI.csv || fail "Failed to insert column names"
rm ConfLongDemo_JSI.txt || fail "Failed to delete file"
cd .. || fail "Failed to go back"
