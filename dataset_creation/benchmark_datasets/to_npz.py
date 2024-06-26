#%%
import dgl
from pathlib import Path
import torch
from typing import List
import openmm
import openff.toolkit
import json
import numpy as np
from grappa.utils.openmm_utils import get_energies, remove_forces_from_system
from grappa.utils.openff_utils import get_peptide_system

from grappa.data import MolData

from openmm.unit import Quantity
from openmm.unit import mole, hartree, bohr, angstrom, kilocalories_per_mole

import matplotlib.pyplot as plt




#%%

def load_graph(molpath):
    [g], _ = dgl.load_graphs(str(molpath / "heterograph.bin"))
    return g

def load_mol(molpath):

    with open(str(molpath / "mol.json"), 'r') as file:
        moldata = json.load(file)
        # convert from str to dict:
        moldata = json.loads(moldata)
    if not 'partial_charge_unit' in moldata.keys():
        moldata['partial_charge_unit'] = moldata['partial_charges_unit']
    if "hierarchy_schemes" not in moldata.keys():
        moldata["hierarchy_schemes"] = dict()

    mol = openff.toolkit.topology.Molecule.from_dict(moldata)
    return mol
#%%
def extract_data(g, mol):
    """
    Converts data to grappa units (kcal/mol, Angstrom, elementary charge).
    """

    am1bcc_elf_charges = mol.partial_charges.to_openmm().value_in_unit(openmm.unit.elementary_charge)

    am1bcc_elf_charges = np.array(am1bcc_elf_charges)

    atomic_numbers = np.array([a.atomic_number for a in mol.atoms])

    mapped_smiles = mol.to_smiles(mapped=True)
    smiles = mol.to_smiles()

    ENERGY_UNIT = kilocalories_per_mole
    DISTANCE_UNIT = angstrom
    FORCE_UNIT = ENERGY_UNIT / DISTANCE_UNIT

    PARTICLE = mole.create_unit(6.02214076e23 ** -1, "particle", "particle")
    HARTREE_PER_PARTICLE = hartree / PARTICLE

    esp_force = HARTREE_PER_PARTICLE / bohr
    esp_energy = HARTREE_PER_PARTICLE
    esp_distance = bohr
        


    data = {}
    data['am1bcc_elf_charges'] = am1bcc_elf_charges
    data['atomic_numbers'] = atomic_numbers
    data['mapped_smiles'] = np.array([mapped_smiles])
    data['smiles'] = np.array([smiles])

    data['xyz'] = g.nodes['n1'].data['xyz'].transpose(0,1).numpy()

    data['xyz'] = Quantity(data['xyz'], esp_distance).value_in_unit(DISTANCE_UNIT)

    assert data['xyz'].shape[1] == len(data['am1bcc_elf_charges']) == len(data['atomic_numbers'])

    ff_names = ['qm', 'ref', 'openff-2.0.0', 'openff-1.2.0', 'gaff-2.11']
    if 'u_amber14' in g.nodes['g'].data.keys():
        ff_names.append('amber14')

    for ff_name in ff_names:

        # we store the data in units of kcal/mol, Angstrom and elementary charges. in all arrays, axis order is:
        # conformation, atom, spatial dimension
        # i.e. forces are stored as (conformation, atom, spatial dimension), atomic numbers as (atom)

        data[f'energy_{ff_name}'] = g.nodes['g'].data[f'u_{ff_name}'][0].numpy()
        data[f'gradient_{ff_name}'] = g.nodes['n1'].data[f'u_{ff_name}_prime'].transpose(0,1).numpy()

        assert len(data[f'energy_{ff_name}'].shape) == 1
        assert len(data[f'gradient_{ff_name}'].shape) == 3

        assert data[f'energy_{ff_name}'].shape[0] == data[f'gradient_{ff_name}'].shape[0] == data['xyz'].shape[0]
        assert data[f'gradient_{ff_name}'].shape == data['xyz'].shape

        # convert to angstrom and kcal/mol
        data[f'energy_{ff_name}'] = Quantity(data[f'energy_{ff_name}'], esp_energy).value_in_unit(ENERGY_UNIT)
        data[f'gradient_{ff_name}'] = Quantity(data[f'gradient_{ff_name}'], esp_force).value_in_unit(FORCE_UNIT)

    return data


# %%


def main(dspath, targetpath, with_amber99: bool = True, exclude_pattern: List[str] = None, with_charmm36: bool = False):
    print(f"Converting\n{dspath}\nto\n{targetpath}")
    dspath = Path(dspath)
    targetpath = Path(targetpath)

    targetpath.mkdir(exist_ok=True, parents=True)

    # iterate over all child directories of dspath:
    num_total = 0
    num_success = 0
    num_err = 0

    total_mols = 0
    total_confs = 0

    for idx, molpath in enumerate(dspath.iterdir()):
        if not molpath.is_dir():
            continue
        num_total += 1
        try:
            print(f"Processing {idx}", end='\r')
            g, mol = load_graph(molpath), load_mol(molpath)
            data = extract_data(g, mol)

            if exclude_pattern is not None:
                if any([p in data['smiles'][0] for p in exclude_pattern]):
                    print(f"Excluding {data['smiles'][0][:20]}...")
                    continue

            if with_amber99 or with_charmm36:

                assert not (with_amber99 and with_charmm36), "Can only compute one of amber99sbildn and charmm36 energies and forces!"

                if with_amber99:
                    system = get_peptide_system(mol=mol, ff='amber99sbildn.xml')
                    tag = 'amber99'
                elif with_charmm36:
                    system = get_peptide_system(mol=mol, ff='charmm/toppar_all36_prot_model.xml')
                    tag = 'charmm36'
                    

                # get list or residue names per atom from the system:

                energy_amber99, force_amber99 = get_energies(openmm_system=system, xyz=data['xyz'])

                system = remove_forces_from_system(system=system, keep='nonbonded')

                energy_amber99_nonbonded, force_amber99_nonbonded = get_energies(openmm_system=system, xyz=data['xyz'])

                data[f'energy_{tag}'] = energy_amber99
                data[f'gradient_{tag}'] = -force_amber99
                data[f'energy_{tag}_nonbonded'] = energy_amber99_nonbonded
                data[f'gradient_{tag}_nonbonded'] = -force_amber99_nonbonded

                data['energy_ref'] = data['energy_qm'] - data[f'energy_{tag}_nonbonded']
                data['gradient_ref'] = data['gradient_qm'] - data[f'gradient_{tag}_nonbonded']

                # create moldata from amber99sbildn system
                moldata = MolData.from_openmm_system(openmm_system=system, openmm_topology=mol.to_topology().to_openmm(), mol_id=data['smiles'][0], partial_charges=None, xyz=data['xyz'], energy=data['energy_qm'], gradient=data['gradient_qm'], energy_ref=data['energy_ref'], gradient_ref=data['gradient_ref'], mapped_smiles=data['mapped_smiles'][0], smiles=data['smiles'][0], allow_nan_params=True, charge_model='amber99')

                # add classical ff information
                moldata.ff_energy.update({k.split('_', 1)[1]: v for k, v in data.items() if k.startswith('energy_') and not k == 'energy_ref'})
                moldata.ff_gradient.update({k.split('_', 1)[1]: v for k, v in data.items() if k.startswith('gradient_') and not k == 'gradient_ref'})
                moldata.ff_nonbonded_energy.update({k.split('_', 2)[2]: v for k, v in data.items() if k.startswith('nonbonded_energy_')})
                moldata.ff_nonbonded_gradient.update({k.split('_', 2)[2]: v for k, v in data.items()if k.startswith('nonbonded_gradient_')})


                moldata.save(targetpath/(molpath.stem+'.npz'))
                total_mols += 1
                total_confs += data['xyz'].shape[0]
                num_success += 1
                continue

            total_mols += 1
            total_confs += data['xyz'].shape[0]


            np.savez_compressed(targetpath/(molpath.stem+'.npz'), **data)
            num_success += 1
        except Exception as e:
            raise
            num_err += 1
            print(f"Failed to process {molpath}: {e}")
            continue
    
    print("\nDone!")
    print(f"Processed {num_total} molecules, {num_success} successfully, {num_err} with errors")

    print(f"Total mols: {total_mols}, total confs: {total_confs}")

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dspath",
        type=str,
        default="/hits/fast/mbm/seutelf/esp_data/spice-dipeptide",
        help="Path to the folder with heterograph and mol files from espaloma.",
    )
    parser.add_argument(
        "--targetpath",
        type=str,
        default="/hits/fast/mbm/seutelf/data/datasets/spice-dipeptide",
        help="Path to the target folder in which tha dataset is stored as collection of npz files.",
    )
    parser.add_argument(
        "--with_amber99",
        action="store_true",
        help="Whether to compute amber99sbildn energies and forces and rather use them as reference. Can only be done for peptides. If True, grappa.data.MolData objects will be created and stored directly!",
    )
    parser.add_argument(
        "--with_charmm36",
        action="store_true",
        help="Whether to compute charmm36 energies and forces and rather use them as reference. Can only be done for peptides. If True, grappa.data.MolData objects will be created and stored directly!",
    )
    parser.add_argument(
        "--exclude_pattern",
        type=str,
        nargs='+',
        default=None,
        help="If given, exclude all molecules whose smiles contain this pattern.",
    )
    args = parser.parse_args()
    main(dspath=args.dspath, targetpath=args.targetpath, with_amber99=args.with_amber99, exclude_pattern=args.exclude_pattern, with_charmm36=args.with_charmm36)