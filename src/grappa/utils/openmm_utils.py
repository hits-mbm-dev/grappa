OPENMM_WATER_RESIDUES = ["HOH", "WAT", "TIP3", "TIP4", "TIP5", "TIP3P", "TIP4P", "TIP5P", "SPC", "SPC/E", "SPCE", "SPC-FW", "SPC-HW", "SPC-HFW", "SPC-HF"]
OPENMM_ION_RESIDUES = ["CL", "NA", "K", "MG", "CA", "ZN", "FE", "CU", "F", "BR", "I", "CL-", "NA+", "K+", "MG2+", "CA2+", "ZN2+", "FE2+", "FE3+", "CU2+", "CU1+", "F-", "BR-", "I-"]


import importlib.util
if importlib.util.find_spec('openmm') is not None:
    
    import openmm
    import numpy as np
    from typing import Union, Dict, List
    from pathlib import Path
    import tempfile
    from grappa.constants import get_grappa_units_in_openmm
    from grappa import units
    from typing import Tuple
    import grappa.data



    def get_subtopology(topology:openmm.app.Topology, exclude_residues:List[str]=None)->'openmm.Topology':
        """
        Returns a sub-topology of the given topology, excluding certain residues with names given in exclude_residues.
        The atom.id of the atoms in the sub-topology is the same as the atom.index in the original topology.
        """

        assert isinstance(topology, openmm.app.Topology), f"Expected openmm.app.Topology, but got {type(topology)}"

        if exclude_residues is None:
            return topology
        
        # create a new topology:
        new_topology = openmm.app.Topology()

        new_topol_idx = {} # maps the old atom index to the new atom index

        # add a dummy chain and residue:
        new_chain = new_topology.addChain()
        new_residue = new_topology.addResidue('DUM', new_chain)

        # add all atoms ensuring that their atom.id is the index in the original topology
        for atom in topology.atoms():
            if atom.residue.name not in exclude_residues:
                new_topology.addAtom(atom.name, atom.element, new_residue, id=atom.index)
                new_topol_idx[atom.index] = new_topology.getNumAtoms() - 1

        new_atoms = list(new_topology.atoms())

        # add all bonds:
        # we only add bonds where both atoms are in the new topology
        # obtain the old indices, map to new indices, and pick the atoms from the new topology
        for bond in topology.bonds():
            if all([atom.index in new_topol_idx.keys() for atom in bond]):
                new_topology.addBond(new_atoms[new_topol_idx[bond[0].index]], new_atoms[new_topol_idx[bond[1].index]])

        return new_topology


    def get_energies(openmm_system: openmm.System, xyz:np.ndarray)->Tuple[np.ndarray, np.ndarray]:
        """
        Returns enegries, forces. in units kcal/mol and kcal/mol/angstroem
        Assume that xyz is in angstroem and has shape (num_confs, num_atoms, 3).
        """
        import openmm
        from openmm import unit

        assert len(xyz.shape) == 3, f"xyz must have shape (num_confs, num_atoms, 3), but got {xyz.shape}"
        assert xyz.shape[1] == openmm_system.getNumParticles(), f"Number of atoms in xyz ({xyz.shape[1]}) does not match number of atoms in system ({openmm_system.getNumParticles()})"
        assert xyz.shape[2] == 3, f"xyz must have shape (num_confs, num_atoms, 3), but got {xyz.shape}"

        if xyz.shape[0] == 0:
            return np.array([]).astype(np.float32), np.zeros(xyz.shape).astype(np.float32)

        # create a context:
        integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
        context = openmm.Context(openmm_system, integrator)

        energies = []
        forces = []
        # set positions:
        for pos in xyz:
            context.setPositions(unit.Quantity(pos, unit.angstrom))
            state = context.getState(getEnergy=True, getForces=True)
            energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
            forces_ = state.getForces(asNumpy=True).value_in_unit(unit.kilocalories_per_mole/unit.angstrom)
            energies.append(energy)
            forces.append(forces_)

        return np.array(energies), np.array(forces)


    def remove_forces_from_system(system:openmm.System, remove:Union[List[str], str]=None, keep=None, info=False)->'openmm.System':
        """
        Modifies the OpenMM system by removing forces according to the 'remove' and 'keep' lists.
        Forces are identified by their class name. E.g. to remove all nonbonded forces, use remove='nonbonded', to only keep nonbonded forces, use keep='nonbonded'.

        Parameters:
        - system: The OpenMM System object to modify.
        - remove: A list of strings. Forces with class names containing any of these strings will be removed.
        - keep: A list of strings. If not None, only forces with class names containing these strings will be kept.

        Returns:
        - system: The modified OpenMM System object.
        """

        if not isinstance(remove, list):
            remove = [remove]

        # First, identify the indices of the forces to remove
        forces_to_remove = []
        for i, force in enumerate(system.getForces()):
            force_name = force.__class__.__name__.lower()
            if keep is not None:
                if not any([k.lower() in force_name for k in keep]):
                    forces_to_remove.append(i)
                    if info:
                        print(f"Removing force {force_name}")
            elif remove is not None:
                if any([k.lower() in force_name for k in remove]):
                    forces_to_remove.append(i)
                    if info:
                        print(f"Removing force {force_name}")

        # Remove the forces by index, in reverse order to not mess up the indices
        for i in reversed(forces_to_remove):
            system.removeForce(i)

        return system


    def set_partial_charges(system:openmm.System, partial_charges:Union[list, np.ndarray])->'openmm.System':
        """
        Set partial charges of a system. The charge must be in units of elementary charge.
        """
        import openmm

        # get the nonbonded force (behaves like a reference not a copy!):
        nonbonded_force = None
        for force in system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                if not nonbonded_force is None:
                    raise ValueError("More than one nonbonded force found.")
                nonbonded_force = force
                
        if nonbonded_force is None:
            raise ValueError("No nonbonded force found.")
        
        # set the charges:
        if len(partial_charges) != nonbonded_force.getNumParticles():
            raise ValueError("Number of partial charges does not match number of particles.")
        
        for i, charge in enumerate(partial_charges):
            # get the parameters:
            _, sigma, epsilon = nonbonded_force.getParticleParameters(i)
            # set the charge:
            nonbonded_force.setParticleParameters(i, charge=charge, sigma=sigma, epsilon=epsilon)

        return system


    def write_to_system(system:openmm.System, parameters:grappa.data.Parameters)->'openmm.System':
        """
        Writes bonded parameters in an openmm system. For interactions that are already present in the system, overwrite the parameters; otherwise add the interaction to the system. The forces, however, must be already present in the system.
        The ids of the atoms, bonds, etc in the parameters object must be the same as the system indices.
        The ids must n0t necessarily run from 0 to N-1, they can also represent a subset of the system indices.
        """

        # handle units:
        import openmm
        from openmm.unit import Quantity

        grappa_units = get_grappa_units_in_openmm()
        BOND_K_UNIT = grappa_units['BOND_K']
        BOND_EQ_UNIT = grappa_units['BOND_EQ']
        ANGLE_K_UNIT = grappa_units['ANGLE_K']
        ANGLE_EQ_UNIT = grappa_units['ANGLE_EQ']
        TORSION_K_UNIT = grappa_units['TORSION_K']
        TORSION_PHASE_UNIT = grappa_units['TORSION_PHASE']


        bonds = parameters.bonds
        angles = parameters.angles
        impropers = parameters.impropers
        propers = parameters.propers


        bond_ks = Quantity(parameters.bond_k, unit=BOND_K_UNIT)
        bond_eqs = Quantity(parameters.bond_eq, unit=BOND_EQ_UNIT)
        angle_ks = Quantity(parameters.angle_k, unit=ANGLE_K_UNIT)
        angle_eqs = Quantity(parameters.angle_eq, unit=ANGLE_EQ_UNIT)
        improper_ks = Quantity(parameters.improper_ks, unit=TORSION_K_UNIT)
        improper_phases = Quantity(parameters.improper_phases, unit=TORSION_PHASE_UNIT)
        proper_ks = Quantity(parameters.proper_ks, unit=TORSION_K_UNIT)
        proper_phases = Quantity(parameters.proper_phases, unit=TORSION_PHASE_UNIT)

        assert np.all(parameters.proper_ks >= 0)
        assert np.all(parameters.improper_ks >= 0)

        # create a dictionary because we will need lookups and dict lookup is more efficient than list.index (these are shallow copies)
        bond_lookup = {tuple(b):(bond_ks[i], bond_eqs[i]) for i, b in enumerate(bonds)}
        angle_lookup = {tuple(a): (angle_ks[i], angle_eqs[i]) for i, a in enumerate(angles)}

        ordered_torsions = {tuple(sorted(p)) for p in list([imp for imp in impropers]) + list([prop for prop in propers])}

        # loop through the system forces, for all parameters in the parameters object, overwrite the system parameters if present, otherwise add the interaction.
        # Note that if the system contains bonds/angles/... between atoms that are not in the parameters object, these bonds/angles... will be untouched, i.e. kept as they are.
        # in each step, first transform the parameter id to the system index.

        for force in system.getForces():
            if isinstance(force, openmm.HarmonicBondForce):
                for i in range(force.getNumBonds()):
                    # Get the atom indices and existing parameters
                    atom1, atom2, length, k = force.getBondParameters(i)

                    # try both orderings:
                    bond_param = bond_lookup.get((atom1, atom2), None)
                    if bond_param is None:
                        bond_param = bond_lookup.get((atom2, atom1), None)
                        if not bond_param is None:
                            bond_lookup.pop((atom2, atom1))
                    else:
                        bond_lookup.pop((atom1, atom2))

                    if not bond_param is None:
                        # Update the parameters
                        new_k, new_length = bond_param
                        force.setBondParameters(i, atom1, atom2, new_length, new_k)


            elif isinstance(force, openmm.HarmonicAngleForce):
                for i in range(force.getNumAngles()):
                    atom1, atom2, atom3, _, _ = force.getAngleParameters(i)

                    angle_param = angle_lookup.pop((atom1, atom2, atom3), None)
                    if angle_param is None:
                        angle_param = angle_lookup.pop((atom3, atom2, atom1), None)

                    if not angle_param is None:
                        new_k, new_angle = angle_param
                        force.setAngleParameters(i, atom1, atom2, atom3, new_angle, new_k)


            # check whether torsion is contained in both proper or improper. if so, set its k to zero, effectively removing the force.
            if isinstance(force, openmm.PeriodicTorsionForce):
                for i in range(force.getNumTorsions()):
                    atom1, atom2, atom3, atom4, periodicity, phase, k = force.getTorsionParameters(i)

                    # Check in proper and improper torsions
                    if tuple(sorted((atom1, atom2, atom3, atom4))) in ordered_torsions:
                        # Set k to zero to effectively remove from the system. We will add another torsion force later.
                        force.setTorsionParameters(i, atom1, atom2, atom3, atom4, periodicity, phase, 0)

        
            # now add the bonds and angles that have not been added yet as new forces.
            # also add a new torsion force, one for proper, one for improper.

        # Adding remaining bonds
        if bond_lookup:
            new_bond_force = openmm.HarmonicBondForce()
            for bond, params in bond_lookup.items():
                new_bond_force.addBond(bond[0], bond[1], length=params[1], k=params[0])
            system.addForce(new_bond_force)

        # Adding remaining angles
        if angle_lookup:
            new_angle_force = openmm.HarmonicAngleForce()
            for angle, params in angle_lookup.items():
                new_angle_force.addAngle(angle[0], angle[1], angle[2], angle=params[1], k=params[0])
            system.addForce(new_angle_force)

        # Adding all torsions:
        proper_torsion_force = openmm.PeriodicTorsionForce()
        for i, torsion in enumerate(propers):
            for n in range(len(proper_ks[i])):
                if proper_ks[i][n].value_in_unit(TORSION_K_UNIT) != 0.:
                    proper_torsion_force.addTorsion(torsion[0], torsion[1], torsion[2], torsion[3], periodicity=n+1, phase=proper_phases[i][n], k=proper_ks[i][n])

        # Adding all impropers:
        improper_torsion_force = openmm.PeriodicTorsionForce()
        for i, torsion in enumerate(impropers):
            for n in range(len(improper_ks[i])):
                if improper_ks[i][n].value_in_unit(TORSION_K_UNIT) != 0.:
                    improper_torsion_force.addTorsion(torsion[0], torsion[1], torsion[2], torsion[3], periodicity=n+1, phase=improper_phases[i][n], k=improper_ks[i][n])

        system.addForce(proper_torsion_force)
        system.addForce(improper_torsion_force)

        return system



    def topology_from_pdb(pdbstring:str)->'openmm.Topology':
        """
        Returns an openmm topology from a pdb string in which the lines are separated by '\n'.
        """
        from openmm.app import PDBFile

        with tempfile.TemporaryDirectory() as tmp:
            pdbpath = str(Path(tmp)/'pep.pdb')
            with open(pdbpath, "w") as pdb_file:
                pdb_file.write(pdbstring)
            openmm_pdb = PDBFile(pdbpath)

        return openmm_pdb.topology


    def get_openmm_forcefield(name:str, *args, **kwargs):
        """
        The name can be given either with or without .xml ending. Possible names are all openmm forcefield names and:
        - amber99sbildn* or amber99sbildn-star (amber99sbildn with HYP and DOP)
        """
        from openmm.app import ForceField

        if name.endswith('.xml'):
            name = name[:-4]
        
        if name == 'amber99sbildn*' or name == 'amber99sbildn-star':
            from grappa.utils import hyp_dop_utility

            ff_path = Path(__file__).parent / Path("amber99sbildn-star_.xml")

            class HypDopOpenmmForceField:
                """
                Modify the createSystem method because openmm.PDBFile cannot read HYP and DOP residues properly.
                """
                def __init__(self, ff_path:Union[Path, str], *args, **kwargs):
                    self.ff = ForceField(str(ff_path), *args, **kwargs)
                
                def createSystem(self, topology, *args, **kwargs):
                    """
                    Create the system. This method is overwritten because openmm.PDBFile cannot read HYP and DOP residues properly.
                    """
                    # add bonds that were not written to the topology yet:

                    topology = hyp_dop_utility.add_bonds(topology)
                    return self.ff.createSystem(topology, *args, **kwargs)
                    
            return HypDopOpenmmForceField(str(ff_path), *args, **kwargs)

        else:
            return ForceField(name+'.xml')
