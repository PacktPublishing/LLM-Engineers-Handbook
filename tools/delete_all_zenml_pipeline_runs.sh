#!/bin/bash

# Extract all the RUN IDs into a temporary file
zenml pipeline runs list | grep -oE '[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}' > run_ids.txt

# Iterate over the run_ids.txt file and delete each run
while read -r RUN_ID; do
    echo "Deleting pipeline run with ID: $RUN_ID"
    zenml pipeline runs delete "$RUN_ID"
done < run_ids.txt

# Optionally, remove the temporary run_ids.txt file
rm run_ids.txt

echo "All specified pipeline runs have been deleted."
