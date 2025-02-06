#!/bin/bash

# This script executes the Python test script located in the 'tests' directory.
# It also creates the Conda environment using the environment.yml file if needed.

# D'ont forget to execute chmod +x run_test_script.sh

# Print a message indicating the script is starting
echo "Starting the process..."

# Check if the Conda environment exists, if not, create it from the .yml file
if ! conda env list | grep -q "Giskard_test"; then
    echo "Creating Conda environment from environment.yml file..."
    conda env create -f environment_Giskard.yml
else
    echo "Conda environment already exists."
fi

# Activate your Conda environment
conda activate Giskard_test

# Execute the Python test script
python -m tests.test_script

# Check if the command ran successfully
if [ $? -eq 0 ]; then
    echo "Test script executed successfully!"
else
    echo "Test script execution failed."
fi

# Deactivate the Conda environment
conda deactivate
