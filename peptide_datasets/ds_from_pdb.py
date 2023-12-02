"""
Parametrise the spice dipeptide dataset with the Amber 99SBildn forcefield and store the dataset as npz files representing a MolData object.
"""


from grappa.data import MolData, Molecule
from pathlib import Path
import numpy as np
from openmm.app import ForceField
import tempfile
from openmm.app import PDBFile
from grappa.utils import openmm_utils

def main(source_path, target_path, forcefield):
    print(f"Converting\n{source_path}\nto\n{target_path}")
    source_path = Path(source_path)
    target_path = Path(target_path)

    target_path.mkdir(exist_ok=True, parents=True)

    # iterate over all child directories of source_path:
    num_total = 0
    num_success = 0
    num_err = 0

    total_mols = 0
    total_confs = 0

    for idx, molfile in enumerate(source_path.iterdir()):
        if molfile.is_dir():
            continue
        num_total += 1
        try:
            print(f"Processing {idx}", end='\r')
            data = np.load(molfile)
            # transform to actual dictionary
            data = {k:v for k,v in data.items()}

            xyz = data['n1 xyz'].transpose(1,0,2)
            gradient = data['n1 grad_qm'].transpose(1,0,2)
            energy = data['g u_qm'][0]
            pdb = data['pdb'].tolist()
            pdbstring = ''.join(pdb)
            sequence = str(data['sequence'])

            # get topology:
            topology = openmm_utils.topology_from_pdb(pdbstring)

            system = ForceField(forcefield).createSystem(topology)

            total_mols += 1
            total_confs += xyz.shape[0]

            # create moldata object from the system (calculate the parameters, nonbonded forces and create reference energies and gradients from that)
            moldata = MolData.from_openmm_system(openmm_system=system, openmm_topology=topology, xyz=xyz, gradient=gradient, energy=energy, mol_id=sequence, pdb=pdbstring)

            moldata.molecule.add_features(['ring_encoding'])

            moldata.save(target_path/(molfile.stem+'.npz'))

            num_success += 1
        except Exception as e:
            num_err += 1
            raise
            # print(f"Failed to process {molpath}: {e}")
            continue
    
    print("\nDone!")
    print(f"Processed {num_total} molecules, {num_success} successfully, {num_err} with errors")

    print(f"Total mols: {total_mols}, total confs: {total_confs}")

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_path",
        type=str,
        help="Path to the folder with npz files containing smiles, positions, energies and gradients.",
    )
    parser.add_argument(
        "--target_path",
        type=str,
        help="Path to the target folder in which the dataset is stored as collection of npz files.",
    )
    parser.add_argument(
        "--forcefield",
        type=str,
        default='amber99sbildn.xml',
        help="Which forcefield to use for creating improper torsion and classical parameters. if no energy_ref and gradient_ref are given, the nonbonded parameters are used as reference.",
    )
    args = parser.parse_args()
    main(source_path=args.source_path, target_path=args.target_path, forcefield=args.forcefield)