#!/bin/bash

# throw upon error:
set -e

# grappa provides a comand-line application ('grappa_gmx') that creates gromacs topology files.
# The workflow is as follows:
# 1. Create gromacs topology file using a classical forcefield
# 2. Run grappa_gmx path/to/topology.top path/to/new_topology.top --modeltag modeltag
# 3. Proceed to use the new topology file in gromacs

# Example:
# First, create a gromacs topology file using the amber99sbildn forcefield
mkdir -p mdrun
pushd mdrun

# (the 6 1 flags are to select the traditional forcefield and water model)
printf "6\n1\n "|gmx pdb2gmx -f ../T4.pdb -o T4.gro -p T4.top -ignh

# Then, run grappa_gmx to create a new topology file using the grappa model
# (This is the only line that depends on grappa, the rest is standard gromacs usage.)
# the -p flag is to create a plot of grappa's predicted parameters
############################################
grappa_gmx -f T4.top -o T4_grappa.top -t grappa-1.2.2 -p
############################################


# Finally, use the new topology file in gromacs, e.g. in an energy minimization
gmx editconf -f T4.gro -o T4_box.gro -c -d 1.0 -bt dodecahedron
gmx solvate -cp T4_box.gro -p T4_grappa.top -o T4_solv.gro

gmx grompp -f ions.mdp -c T4_solv.gro -p T4_grappa.top -o T4_out_genion.tpr
echo "SOL" | gmx genion -s T4_out_genion.tpr -p T4_grappa.top -o T4_out_ion.gro -neutral

gmx grompp -f minim.mdp -c T4_out_ion.gro -p T4_grappa.top -o T4_out_min.tpr
gmx mdrun -deffnm T4_out_min -v

popd