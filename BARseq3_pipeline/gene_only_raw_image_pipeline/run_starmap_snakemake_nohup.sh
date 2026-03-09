#!/bin/bash

# Path to the JSON file
config_file="config.json"

# Extract values from the JSON file
data_path=$(grep '"path"' "$config_file" | sed 's/.*: "\(.*\)",/\1/')
gtca_path=$(grep '"gtca_path"' "$config_file" | sed 's/.*: "\(.*\)",/\1/')
section=$(grep '"section"' "$config_file" | sed 's/.*: "\(.*\)",/\1/')

# Concatenate data_path and gtca_path
full_gtca_path="${data_path}${gtca_path}"

# Copy snakemake file to data_path
#cp Snakefile_nopatch "$data_path"

# Use the concatenated path in the Snakemake command
nohup snakemake --slurm --jobs "$(python -c "import h5py; f=h5py.File('$full_gtca_path', 'r'); print(int(len(f['list'])))")" \
   -s "$data_path"Snakefile_nopatch \
   --configfile "$config_file" \
   --directory "$data_path" \
   --profile profile \
   --latency-wait 300 \
   --rerun-incomplete > snakemake_output_$section.log 2>&1 &
