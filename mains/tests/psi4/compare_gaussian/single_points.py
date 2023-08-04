from ase import Atoms
from ase.calculators.psi4 import Psi4
from ase.io import read, write
from ase import units as ase_units
import numpy as np

# import openmm as mm
from pathlib import Path


###################
import sys
import os

class no_print:
    """
    Context manager to suppress stdout.
    """
    def __enter__(self):
        # Save the original stdout
        self.original_stdout = sys.stdout 
        # Set stdout to null device
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Reset stdout to original
        sys.stdout = self.original_stdout
###################


def calc_states(pdb_folder, n_states=None):

    METHOD = 'bmk'
    BASIS = '6-311+G(2df,p)'
    MEMORY = '20GB'
    NUM_THREADS=10


    pdb_folder = Path(pdb_folder)

    if not (pdb_folder/Path("positions.npy")).exists():
        return

    positions = np.load(str(pdb_folder/Path("positions.npy")))
    atomic_numbers = np.load(str(pdb_folder/Path("atomic_numbers.npy")))
    


  
    # Calculate energies and forces using Psi4
    psi4_energies = []
    psi4_forces = []
    
    # load if present:
    if (pdb_folder/Path("psi4_energies.npy")).exists() and (pdb_folder/Path("psi4_forces.npy")).exists():
        psi4_energies = [e for e in np.load(str(pdb_folder/Path("psi4_energies.npy")))]
        psi4_forces = [f for f in np.load(str(pdb_folder/Path("psi4_forces.npy")))]
        

    from time import time
    start = time()

    missing_indices = range(len(psi4_energies), len(positions))

    # restroct the amount of calculations:
    if n_states is not None:
        if len(missing_indices) > n_states:
            missing_indices = missing_indices[:n_states]

    for num_calculated, i in enumerate(missing_indices):
        
        msg = f"calculating state number {i}/{len(positions)-1}... Progress: {num_calculated}/{len(missing_indices)-1}, time elapsed: {round((time() - start)/60., 2)} min"
        if i > 0:
            msg += f", avg time per state: {round((time() - start)/(i) / 60.,2)} min"
        print(msg)

        # Read the configuration
        atoms = Atoms(numbers=atomic_numbers, positions=positions[i])

        atoms.set_calculator(Psi4(atoms=atoms, method=METHOD, memory=MEMORY, basis=BASIS, num_threads=NUM_THREADS))

        energy = atoms.get_potential_energy(apply_constraint=False) # units: eV
        forces = atoms.get_forces(apply_constraint=False) # units: eV/Angstrom

        EV_IN_KCAL = 23.0609

        energy = energy * EV_IN_KCAL
        forces = forces * EV_IN_KCAL


        psi4_energies.append(energy)
        psi4_forces.append(forces)

        # save the energies and forces in every step:
        np.save(str(pdb_folder/Path("psi4_energies.npy")), np.array(psi4_energies))
        np.save(str(pdb_folder/Path("psi4_forces.npy")), np.array(psi4_forces))




def calc_all_states(folder, n_states=None, skip_errs=False):
    from pathlib import Path
    for i, pdb_folder in enumerate(Path(folder).iterdir()):
        if pdb_folder.is_dir():
            print()
            print(f"generating states for {i}")
            # calc_states(pdb_folder)
            try:
                calc_states(pdb_folder, n_states=n_states)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                if not skip_errs:
                    raise

                print(f"failed to generate states for {i} ({Path(folder).stem}): {type(e)}\n: {e}")
                # raise
                pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate states for a given folder.')
    parser.add_argument('folder', type=str, help='The folder containing the PDB files.')
    parser.add_argument('--n_states', '-n', type=int, help='The number of states to generate.', default=None)
    parser.add_argument('--skip_errs', '-s', action='store_true', help='Skip errors.', default=False)
    args = parser.parse_args()
    calc_all_states(folder=args.folder, n_states=args.n_states, skip_errs=args.skip_errs)