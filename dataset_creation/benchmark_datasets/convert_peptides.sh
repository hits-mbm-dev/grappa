#!/bin/bash
# create peptide datasets with amber99sbildn energies and force as reference

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )" # dir in which this script lies

espaloma_ds_path="$SCRIPT_DIR/../../data/esp_data"
target_path="$SCRIPT_DIR/../../data/grappa_datasets" # due to with_amber flag, this will be stored as moldata directly

# python unmerge_duplicates.py --duplpath "$espaloma_ds_path/duplicated-isomeric-smiles-merge" --targetpath "$espaloma_ds_path"

# List of dataset names
datasets=("spice-dipeptide" "pepconf-dlc" "protein-torsion")

# Loop through each dataset name
for ds in "${datasets[@]}"; do
  python to_npz.py --dspath "$espaloma_ds_path/$ds" --targetpath "$target_path/$ds""_amber99sbildn" --with_amber99
done
