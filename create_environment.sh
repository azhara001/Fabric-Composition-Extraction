# create_environment.sh

# Create Conda environment
conda env create -f environment.yml

# Activate Conda environment
conda activate fabric-composition-env

# Verify that the environment was activated
echo "Conda environment activated."

# Install any additional packages or perform other setup steps if needed
# For example:
# conda install -c conda-forge additional_package

# Deactivate Conda environment

echo "Setup complete."
