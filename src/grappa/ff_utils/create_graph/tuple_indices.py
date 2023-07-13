

import numpy as np
from . import utils

from typing import Tuple, Set, Dict, Union, List


from rdkit.Chem.rdchem import Mol

def get_indices(mol: Mol, reduce_symmetry:bool=True) -> Dict:
    """
    Obtain the indices of atoms, bonds, angles, torsions, and impropers in the given molecule.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The molecule to get indices from.
    reduce_symmetry (bool): If True, then index tuples that can be obtained by invariant permutations are removed. Default is True.

    Returns:
    Dict: A dictionary containing the indices of atoms, bonds, angles, torsions, and impropers in the molecule as np.ndarray.
    """
    indices = {}
    indices["n1"] = atom_indices(mol)
    indices["n2"] = bond_indices(mol, reduce_symmetry=reduce_symmetry)
    indices["n3"] = angle_indices(mol, reduce_symmetry=reduce_symmetry)
    
    propers, impropers = torsion_indices(mol, reduce_symmetry=reduce_symmetry)
    indices["n4"] = propers
    indices["n4_improper"] = impropers
    return indices


def atom_indices(mol: Mol) -> np.ndarray:
    """
    Obtain the indices of atoms in the given molecule.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The molecule to get atom indices from.

    Returns:
    np.ndarray: A numpy array containing the indices of atoms in the molecule.
    """
    return np.array([a.GetIdx() for a in mol.GetAtoms()]).astype(np.int64)


def bond_indices(mol: Mol, reduce_symmetry: bool = True) -> np.ndarray:
    """
    Obtain the indices of bonds in the given molecule.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The molecule to get bond indices from.
    reduce_symmetry (bool): If True, then index tuples that can be obtained by invariant permutations are removed. Default is True.

    Returns:
    np.ndarray: A numpy array containing the indices of bonds in the molecule.
    """
    bond_indices = construct_bonds(mol, reduce_symmetry=reduce_symmetry)
    return np.array(list(bond_indices)).astype(np.int64)


def angle_indices(mol: Mol, reduce_symmetry: bool = True) -> np.ndarray:
    """
    Obtain the indices of angles in the given molecule.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The molecule to get angle indices from.
    reduce_symmetry (bool): If True, then index tuples that can be obtained by invariant permutations are removed. Default is True.

    Returns:
    np.ndarray: A numpy array containing the indices of angles in the molecule.
    """
    angle_indices = construct_angles(mol, reduce_symmetry=reduce_symmetry)
    return np.array(list(angle_indices)).astype(np.int64)


def torsion_indices(mol: Mol, reduce_symmetry: bool = True, only_torsion_sets:bool=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Obtain the indices of proper and improper torsions in the given molecule.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The molecule to get torsion indices from.
    reduce_symmetry (bool): If True, then index tuples that can be obtained by invariant permutations are removed. Default is True.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Two numpy arrays containing the indices of proper and improper torsions in the molecule.
    """
    if only_torsion_sets:
        return construct_torsions(mol, reduce_symmetry=reduce_symmetry, only_torsion_sets=only_torsion_sets)
    
    propers, impropers = construct_torsions(mol, reduce_symmetry=reduce_symmetry, only_torsion_sets=only_torsion_sets)
    return np.array(list(propers)).astype(np.int64), np.array(list(impropers)).astype(np.int64)



def construct_bonds(mol, reduce_symmetry:bool=True) -> Set[Tuple]:
    """
    Construct a set containing the index tuples describing bonds
    """
    bonds = set()

    for bond in mol.GetBonds():
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()
        bonds = apply_symmetry(idx_set=bonds, tuple_=tuple((atom1_idx, atom2_idx)), level='n2', reduce_symmetry=reduce_symmetry)
    return bonds


# =============================================================================
# inspired by openff, translated to rdkit molecule:

def construct_torsions(mol, reduce_symmetry:bool=True, central_atom_position:int=3, only_torsion_sets:bool=False)->Union[Tuple[Set[Tuple], Set[Tuple]], List[Set[Tuple]]]:
    """
    Returns propers, impropers
    Construct sets containing the index tuples describing improper and proper torsions
    If reduce_symmetry is True, then index tuples that can be obtained by invariant permutations are removed, if not, they are included.
    central_atom_position is the position of the central atom in the index tuple, i.e. tuple[central_atom_position-1] is the central atom. For amber this is 3 by default.
    only tested for central_atom_position==3.
    If only_toresion_sets is True, then only a set of index sets describing proper torsions is returned. This can help to determine whether a torsion is proper or improper.
    """

    assert central_atom_position == 3, "does not work yet for everything that is not amber since central_atom_position is not implemented for utils.get_symmetric_tuples"

    propers = set()
    impropers = set()

    torsion_sets = []

    for atom1 in mol.GetAtoms():
        atom1_idx = atom1.GetIdx()

        for atom2 in atom1.GetNeighbors():
            atom2_idx = atom2.GetIdx()

            for atom3 in atom2.GetNeighbors():
                atom3_idx = atom3.GetIdx()

                if atom1_idx == atom3_idx:
                    continue

                for atom4 in atom3.GetNeighbors():
                    atom4_idx = atom4.GetIdx()

                    if atom4_idx == atom2_idx:
                        continue
                    # Exclude i-j-k-i
                    if atom1_idx == atom4_idx:
                        continue

                    # =============================================================
                    if only_torsion_sets:
                        if not set([atom1_idx, atom2_idx, atom3_idx, atom4_idx]) in torsion_sets:
                            torsion_sets.append(set([atom1_idx, atom2_idx, atom3_idx, atom4_idx]))
                        continue
                    # =============================================================

                    if atom1_idx < atom4_idx:
                        torsion = (atom1_idx, atom2_idx, atom3_idx, atom4_idx)
                    else:
                        torsion = (atom4_idx, atom3_idx, atom2_idx, atom1_idx)

                    propers = apply_symmetry(idx_set=propers, tuple_=tuple(torsion), level='n4', reduce_symmetry=reduce_symmetry)

                for atom3i in atom2.GetNeighbors():
                    atom3i_idx = atom3i.GetIdx()

                    if atom3i_idx == atom3_idx or atom3i_idx == atom1_idx:
                        continue


                    # rename for convenience:
                    atom4_idx = atom3i_idx

                    # the central atom is atom2_idx, we wish to have it at position central_atom_position, therefore simply swap the indices:
                    improper = [atom1_idx, atom2_idx, atom3_idx, atom4_idx]
                    improper[1], improper[central_atom_position-1] = improper[central_atom_position-1], improper[1]
                    
                    # check if the symmetric version of this improper has already been added
                    impropers = apply_symmetry(idx_set=impropers, tuple_=tuple(improper), level='n4_improper', reduce_symmetry=reduce_symmetry)

    if only_torsion_sets:
        return torsion_sets

    return propers, impropers



def construct_angles(mol, reduce_symmetry:bool=True):
    """
    Get the set of angles.
    If reduce_symmetry is True, then index tuples that can be obtained by invariant permutations are removed, if not, they are included.
    """
    angles = set()

    for atom1 in mol.GetAtoms():
        atom1_idx = atom1.GetIdx()

        for atom2 in atom1.GetNeighbors():
            atom2_idx = atom2.GetIdx()

            for atom3 in atom2.GetNeighbors():
                atom3_idx = atom3.GetIdx()

                if atom1_idx == atom3_idx:
                    continue

                angles = apply_symmetry(idx_set=angles, tuple_=tuple([atom1_idx, atom2_idx, atom3_idx]), level='n3', reduce_symmetry=reduce_symmetry)
                
    return angles

# =============================================================================



def apply_symmetry(idx_set:set, tuple_, level:str, reduce_symmetry:bool)->set:
    """
    Helper function that adds the tuple_ to the idx_set.
    Depending on reduce_symmetry, it either adds all invariant permutations of the tuple_ to the idx_set (reduce_symmetry==False), or only adds the tuple_ if none of the invariant permutations are already in the idx_set (reduce_symmetry==True).
    """
    invariant_tuples = get_symmetric_tuples(level=level, tuple=tuple_)
    if reduce_symmetry:
        # if any of the invariant tuples is already in the set, don't add this improper
        if not any([t in idx_set for t in invariant_tuples]):
            idx_set.add(tuple_)

    else:
        # if all invariant tuples are in the set, continue, else add all (improper is a set, so this is fine)
        if not all([t in idx_set for t in invariant_tuples]):
            [idx_set.add(t) for t in invariant_tuples]

    return idx_set


def get_symmetric_tuples(level, tuple:Union[Tuple[int], List[int]], allow_antisymmetry:bool=True):
    """
    Returns permutations of the index tuple under which the interaction coordinate is invariant. For dihedrals, this is inversion. Permutation of the outer atoms leads to the same interaction coordinate, but with the opposite sign. Since the interaction coordinates are not independent in this case, we can only take one of them when constructing the set of tuples and keep the antisymmetry in mind for the energy function.
    """
    # if level == "n4_improper":
    #     assert len(tuple) == 4, "tuple must be of length 4 for n4_improper"
    #     invariant_tuples = [(tuple[i], tuple[j], tuple[2], tuple[k]) for i,j,k in [(0,1,3), (1,3,0), (3,0,1)]]
    if level == "n4" or level == "n4_improper":
        assert len(tuple) == 4, "tuple must be of length 4 for n4"
        permutations = [(0,1,2,3), (3,2,1,0), (3,1,2,0), (0,2,1,3)]
        invariant_tuples = [(tuple[i], tuple[j], tuple[k], tuple[l]) for i,j,k,l in permutations]
    elif level == "n3":
        assert len(tuple) == 3, "tuple must be of length 3 for n3"
        invariant_tuples = [(tuple[i], tuple[j], tuple[k]) for i,j,k in [(0,1,2), (2,1,0)]]
    elif level == "n2":
        assert len(tuple) == 2, "tuple must be of length 2 for n2"
        invariant_tuples = [(tuple[i], tuple[j]) for i,j in [(0,1), (1,0)]]
    else:
        raise ValueError("Invalid level. Expected one of 'n4_improper', 'n4', 'n3', or 'n2'")

    return invariant_tuples