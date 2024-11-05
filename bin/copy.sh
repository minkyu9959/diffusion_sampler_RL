#!/bin/bash

# Define paths
ORIGINAL_DIR="/home/guest_dyw/diffusion-sampler"

TEMP_DIR=".tmp/$(date +%Y-%m-%d/%H-%M-%S)"

cd ${ORIGINAL_DIR}

# Create temporary directory
mkdir -p ${TEMP_DIR}

# Copy the whole code base to the temporary location
ls ${ORIGINAL_DIR} | grep -P ^\(?\!results\)\(?\!scripts\).*$ | xargs -I {} cp -r --parents {} ${TEMP_DIR}

cd ${TEMP_DIR}

# Run the script
$@

cd ${ORIGINAL_DIR}

rm -r ${TEMP_DIR}
