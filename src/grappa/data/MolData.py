"""
Contains the grappa input dataclass 'MolData', which is an extension of the dataclass 'Molecule' that contains conformational data and characterizations like smiles string or PDB file.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Union
import numpy as np
from grappa.data import Molecule, Parameters
from grappa.utils import openff_utils
import torch
from dgl import DGLGraph

import pkgutil


@dataclass
class MolData():
    """
    Dataclass for entries in datasets on which grappa can be trained.
    """
    molecule: Molecule
    classical_parameters: Parameters # these are used for regularisation and to estimate the statistics of the reference energy and gradient

    # conformational data:
    xyz: np.ndarray
    energy: np.ndarray
    gradient: np.ndarray

    # reference values (centered bonded energy, bonded gradient)
    energy_ref: np.ndarray
    gradient_ref: np.ndarray
    
    # additional characterizations:
    mapped_smiles: Optional[str] = None
    pdb: Optional[str] = None

    # nonbonded contributions:
    ff_nonbonded_energy: Dict[str, np.ndarray] = None
    ff_nonbonded_gradient: Dict[str, np.ndarray] = None

    # classical forcefield energies: dictionaries mapping a force field name to an array of energies
    ff_energy: Dict[str, np.ndarray] = None
    ff_gradient: Dict[str, np.ndarray] = None


    def _validate(self):
        # parameter atoms must be same as molecule atoms
        # check shapes

        pass
    
    def  __post_init__(self):

        # setting {} by default is not possible in dataclasses, thus do this here:
        if self.ff_energy is None:
            self.ff_energy = dict()
        if self.ff_gradient is None:
            self.ff_gradient = dict()
        if self.ff_nonbonded_energy is None:
            self.ff_nonbonded_energy = dict()
        if self.ff_nonbonded_gradient is None:
            self.ff_nonbonded_gradient = dict()

        self._validate()
    
    
    def to_dgl(self, max_element=53, exclude_feats:list[str]=[])->DGLGraph:
        """
        Converts the molecule to a dgl graph with node features. The elements are one-hot encoded.
        Also creates entries 'xyz', 'energy_ref' and 'gradient_ref' in the global node type g and in the atom node type n1, which contain the reference energy and gradient, respectively. The shapes are different than in the class attributes, namely (1, n_confs) and (n_atoms, n_confs, 3) respectively. (This is done because feature tensors must have len == num_nodes)

        The dgl graph has the following node types:
            - g: global
            - n1: atoms
            - n2: bonds
            - n3: angles
            - n4: propers
            - n4_improper: impropers
        The node type n1 carries the feature 'ids', which are the identifiers in self.atoms. The other interaction levels (n{>1}) carry the idxs (not ids) of the atoms as ordered in self.atoms as feature 'idxs'. These are not the identifiers but must be translated back to the identifiers using ids = self.atoms[idxs] after the forward pass.
        This also stores classical parameters except for improper torsions.
        """
        g = self.molecule.to_dgl(max_element=max_element, exclude_feats=exclude_feats)
        
        # write reference energy and gradient in the shape (1, n_confs) and (n_atoms, n_confs, 3) respectively
        g.nodes['g'].data['energy_ref'] = torch.tensor(self.energy_ref.reshape(1, -1), dtype=torch.float32)
        
        g.nodes['g'].data['energy_ref'] -= g.nodes['g'].data['energy_ref'].mean(dim=1)

        g.nodes['n1'].data['gradient_ref'] = torch.tensor(self.gradient_ref.transpose(1, 0, 2), dtype=torch.float32)

        # write positions in shape (n_atoms, n_confs, 3)
        g.nodes['n1'].data['xyz'] = torch.tensor(self.xyz.transpose(1, 0, 2), dtype=torch.float32)

        g = self.classical_parameters.write_to_dgl(g=g)

        return g
    

    def to_dict(self):
        """
        Save the molecule as a dictionary of arrays.
        """
        array_dict = dict()
        array_dict['xyz'] = self.xyz
        array_dict['energy'] = self.energy
        array_dict['gradient'] = self.gradient
        array_dict['energy_ref'] = self.energy_ref
        array_dict['gradient_ref'] = self.gradient_ref

        moldict = self.molecule.to_dict()
        assert set(moldict.keys()).isdisjoint(array_dict.keys()), "Molecule and MolData have overlapping keys."
        array_dict.update(moldict)

        # remove bond, angle, proper, improper since these are stored in the molecule
        paramdict = {
            k: v for k, v in self.classical_parameters.to_dict() if k not in ['bond', 'angle', 'proper', 'improper']
            }

        assert set(paramdict.keys()).isdisjoint(array_dict.keys()), "Parameters and MolData have overlapping keys."

        array_dict.update(paramdict)

        # add force field energies and gradients
        for v, k in self.ff_energy.items():
            if f'energy_{v}' in array_dict.keys():
                raise ValueError(f"Duplicate key: energy_{v}")
            array_dict[f'energy_{v}'] = k
        for v, k in self.ff_gradient.items():
            if f'gradient_{v}' in array_dict.keys():
                raise ValueError(f"Duplicate key: gradient_{v}")
            array_dict[f'gradient_{v}'] = k
        for v, k in self.ff_nonbonded_energy.items():
            if f'nonbonded_energy_{v}' in array_dict.keys():
                raise ValueError(f"Duplicate key: nonbonded_energy_{v}")
            array_dict[f'nonbonded_energy_{v}'] = k
        for v, k in self.ff_nonbonded_gradient.items():
            if f'nonbonded_gradient_{v}' in array_dict.keys():
                raise ValueError(f"Duplicate key: nonbonded_gradient_{v}")
            array_dict[f'nonbonded_gradient_{v}'] = k

        return array_dict
    

    @classmethod
    def from_dict(cls, array_dict:Dict):
        """
        Create a Molecule from a dictionary of arrays.
        """
        xyz = array_dict.get('xyz')
        energy = array_dict.get('energy')
        gradient = array_dict.get('gradient')
        energy_ref = array_dict.get('energy_ref')
        gradient_ref = array_dict.get('gradient_ref')

        # Reconstruct the molecule from the dictionary
        molecule_dict = {k: v for k, v in array_dict.items() if k.startswith('mol_')}
        molecule = Molecule.from_dict(molecule_dict)

        # Reconstruct the parameters, excluding keys that are part of the molecule
        param_keys = ['bond_k', 'bond_eq', 'angle_k', 'angle_eq', 'proper_ks', 'proper_phases', 'improper_ks', 'improper_phases']
        param_dict = {k: array_dict[k] for k in param_keys if k in array_dict}
        classical_parameters = Parameters.from_dict(param_dict)

        # Extract force field energies and gradients
        ff_energy = {k.split('_', 1)[1]: v for k, v in array_dict.items() if k.startswith('energy_') and not k=='energy_ref'}
        ff_gradient = {k.split('_', 1)[1]: v for k, v in array_dict.items() if k.startswith('gradient_') and not k == 'gradient_ref'}
        ff_nonbonded_energy = {k.split('_', 2)[2]: v for k, v in array_dict.items() if k.startswith('nonbonded_energy_')}
        ff_nonbonded_gradient = {k.split('_', 2)[2]: v for k, v in array_dict.items()if k.startswith('nonbonded_gradient_')}

        # Initialize a new MolData object
        return cls(
            xyz=xyz,
            energy=energy,
            gradient=gradient,
            energy_ref=energy_ref,
            gradient_ref=gradient_ref,
            molecule=molecule,
            classical_parameters=classical_parameters,
            ff_energy=ff_energy,
            ff_gradient=ff_gradient,
            ff_nonbonded_energy=ff_nonbonded_energy,
            ff_nonbonded_gradient=ff_nonbonded_gradient
        )

    

    def save(self, path:str):
        """
        Save the molecule to a npz file.
        """
        np.savez(path, **self.to_dict())

    @classmethod
    def load(cls, path:str):
        """
        Load the molecule from a npz file.
        """
        array_dict = np.load(path)
        return cls.from_dict(array_dict)


    @classmethod
    def from_data_dict(cls, data_dict:Dict[str, Union[np.ndarray, str]], forcefield='openff-1.2.0.offxml', partial_charge_key:str='partial_charges'):
        """
        Create a MolData object from a dictionary containing a mapped_smiles string or pdb and arrays of conformations, energies and gradients, but not necessarily the interaction tuples and classical parameters.
        The forcefield is used to obtain the interaction tuples and classical parameters. If a smiles string is used, the forcefield refers to an openff forcefield. If a pdb file is used, the forcefield refers to an openmm forcefield.
        The following dictionray items are required:
            - Either: mapped_smiles: (str) or pdb: (str).startswith(
            - xyz: (n_confs, n_atoms, 3)
            - energy: (n_confs,)
            - gradient: (n_confs, n_atoms, 3)
        """
        smiles = None
        if 'mapped_smiles' in data_dict.keys():
            smiles = data_dict['mapped_smiles']
            if not isinstance(smiles, str):
                smiles = smiles[0]
        pdb = None
        if 'pdb' in data_dict.keys():
            pdb = data_dict['pdb']
            if not isinstance(pdb, str):
                pdb = pdb[0]
        assert smiles is not None or pdb is not None, "Either a smiles string or a pdb file must be provided."
        assert not (smiles is not None and pdb is not None), "Either a smiles string or a pdb file must be provided, not both."

        xyz = data_dict['xyz']
        energy = data_dict['energy_qm']
        gradient = data_dict['gradient_qm']
        partial_charges = data_dict.get(partial_charge_key, None)
        energy_ref = data_dict.get('energy_ref', None)
        gradient_ref = data_dict.get('gradient_ref', None)

        if smiles is not None:
            self = cls.from_smiles(mapped_smiles=smiles, xyz=xyz, energy=energy, gradient=gradient, openff_forcefield=forcefield, partial_charges=partial_charges, energy_ref=energy_ref, gradient_ref=gradient_ref)
        else:
            raise NotImplementedError("pdb files are not supported yet.")



        # Extract force field energies and gradients
        self.ff_energy = {k.split('_', 1)[1]: v for k, v in data_dict.items() if k.startswith('energy_') and not k == 'energy_ref'}
        self.ff_gradient = {k.split('_', 1)[1]: v for k, v in data_dict.items() if k.startswith('gradient_') and not k == 'gradient_ref'}
        self.ff_nonbonded_energy = {k.split('_', 2)[2]: v for k, v in data_dict.items() if k.startswith('nonbonded_energy_')}
        self.ff_nonbonded_gradient = {k.split('_', 2)[2]: v for k, v in data_dict.items()if k.startswith('nonbonded_gradient_')}

        return self


    @classmethod
    def from_openmm_system(cls, openmm_system, openmm_topology, xyz, energy, gradient, partial_charges=None, energy_ref=None, gradient_ref=None, mapped_smiles=None, pdb=None):
        """
        Use an openmm system to obtain classical parameters and interaction tuples.
        If partial charges is None, the charges are obtained from the openmm system.
        If energy_ref and gradient_ref are None, they are also calculated from the openmm system.
        mapped_smiles and pdb have no effect on the system, are optional and only required for reproducibility.
        """
        mol = Molecule.from_openmm_system(openmm_system=openmm_system, openmm_topology=openmm_topology, partial_charges=partial_charges)
        params = Parameters.from_openmm_system(openmm_system, mol=mol)

        self = cls(molecule=mol, classical_parameters=params, xyz=xyz, energy=energy, gradient=gradient, energy_ref=energy_ref, gradient_ref=gradient_ref, mapped_smiles=mapped_smiles, pdb=pdb)

        if self.energy_ref is None:
            # calculate reference energy and gradient from the openmm system using the partial charges provided
            raise NotImplementedError("Reference energy and gradient calculation from openmm system is not yet implemented.")

        return self
    

    @classmethod
    def from_smiles(cls, mapped_smiles, xyz, energy, gradient, partial_charges=None, energy_ref=None, gradient_ref=None, openff_forcefield='openff_unconstrained-1.2.0.offxml'):
        """
        Create a Molecule from a mapped smiles string and an openff forcefield. The openff_forcefield is used to initialize the interaction tuples, classical parameters and, if partial_charges is None, to obtain the partial charges.
        """
        system, topology, _ = openff_utils.get_openmm_system(mapped_smiles, openff_forcefield=openff_forcefield, partial_charges=partial_charges)
        return cls.from_openmm_system(openmm_system=system, openmm_topology=topology, xyz=xyz, energy=energy, gradient=gradient, partial_charges=partial_charges, energy_ref=energy_ref, gradient_ref=gradient_ref, mapped_smiles=mapped_smiles)