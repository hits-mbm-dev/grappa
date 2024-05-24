# creates the grappa 1.1 release and uploads model and datasets to github using github cli.

# throw upon error
set -e

# get an abspath to this script
THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

RUN_ID=leif-seute/grappa-1.2/wm54xocy

MODELNAME=grappa-1.2.2

TAG='v.1.2.0'

# create release
# gh release create $TAG

# cd to this directory:
pushd $THISDIR

# # export model to local models directory
bash prepare_release.sh # NOTE: UNCOMMENT THIS LINE TO EXPORT THE MODEL (will also evaluate, thus it takes some time)
grappa_release -t $TAG -m $MODELNAME

# now upload datasets
# cd to the grappa directory
pushd ../../

# upload files
DATASETS=(
  # spice-des-monomers
  # spice-pubchem
  # gen2
  # gen2-torsion
  # rna-diverse
  # rna-trinucleotide
  # rna-nucleoside
  # spice-dipeptide
  # protein-torsion
  # pepconf-dlc
  spice-dipeptide_amber99sbildn
  # tripeptides_amber99sbildn
  # dipeptide_rad
  # hyp-dop_amber99sbildn
  # AA_bondbreak_rad_amber99sbildn
  # uncapped_amber99sbildn
  # espaloma_split
)

DATADIR=data/dgl_datasets

# for each dir, upload a zipped version of it:
for dir in "${DATASETS[@]}"; do
    echo "Uploading $dir. zipping..."

    # Navigate to the directory just above the target directory
    dir="$DATADIR/$dir"
    parent_dir=$(dirname "$dir")
    dir_name=$(basename "$dir")

    # Change to the parent directory
    pushd "$parent_dir"

    # Create zip file of the directory
    zip -r "$dir_name.zip" "$dir_name"

    # Go back to the original directory
    popd

    # Upload the zip file
    gh release upload $TAG "$parent_dir/$dir_name.zip" --clobber

    # Remove the zip file
    rm "$parent_dir/$dir_name.zip"

done

popd
