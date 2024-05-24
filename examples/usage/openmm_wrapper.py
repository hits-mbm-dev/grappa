#%%
# STANDARD OPENMM WORKFLOW
######################
from pathlib import Path
from openmm.app import ForceField, Topology, PDBFile
from openmm.app import Modeller
from openmm import unit
from grappa import OpenmmGrappa

pdbfile = PDBFile('1ubq.pdb')
topology = pdbfile.topology # load your system as openmm.Topology

classical_ff = ForceField('amber99sbildn.xml', 'tip3p.xml')
# solvate:

modeller = Modeller(topology, pdbfile.positions)
modeller.deleteWater()
modeller.addHydrogens(classical_ff)
modeller.addSolvent(classical_ff, model='tip3p', padding=1.0*unit.nanometers)

topology = modeller.getTopology()
positions = modeller.getPositions()

system = classical_ff.createSystem(topology)
##########################

#%%
# load the pretrained ML model from a tag. Currently, possible tags are grappa-1.1', 'grappa-1.2' and 'latest'
grappa_ff = OpenmmGrappa.from_tag('grappa-1.2.2')

# parametrize the system using grappa. The charge_model tag tells grappa how the charges were obtained, in this case from the classical forcefield amberff99sbildn. possible tags are 'amber99' and 'am1BCC'.
# grappa will not change the solvant parameters and the nonbonded parameters, e.g. the partial charges, Lennard-Jones parameters and combination rules
system = grappa_ff.parametrize_system(system, topology, charge_model='am1BCC', plot_dir='.')

# %%

# SMALL VALIDATION

# now we can use this system downstream. To validate that grappa predicts gradients that are somewhat comparable to those of the classical protein force field, we can plot the gradient components of the grappa system and the original system:

orig_system = classical_ff.createSystem(topology)
from grappa.utils.openmm_utils import get_energies
import numpy as np
from grappa.constants import get_grappa_units_in_openmm

DISTANCE_UNIT = get_grappa_units_in_openmm()['LENGTH']
positions = np.array([positions.value_in_unit(DISTANCE_UNIT)])

# # get energies and gradients of the original system:
orig_energy, original_gradients = get_energies(orig_system, positions)

grappa_energy, grappa_gradients = get_energies(system, positions)

# %%

from matplotlib import pyplot as plt

plt.scatter(original_gradients.flatten(), grappa_gradients.flatten())
plt.xlabel('FF99SBILDN')
plt.ylabel('Grappa')
plt.title('Gradients [kcal/mol/A]')

crmse = np.sqrt(np.mean((original_gradients.flatten() - grappa_gradients.flatten())**2))
plt.text(0.1, 0.9, f'Component RMSE: {crmse:.2f} kcal/mol/A', transform=plt.gca().transAxes)

plt.plot(original_gradients.flatten(), original_gradients.flatten(), color='black', linestyle='--')

plt.savefig('grappa_vs_classical_gradients_T4.png') 
print(f'Component RMSE between Grappa and amber99sbildn: {crmse:.2f} kcal/mol/A')
print('Saved fig to grappa_vs_classical_gradients.png')
# %%
