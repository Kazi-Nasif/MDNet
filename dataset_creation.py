from simtk.openmm.app import *
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import MDAnalysis as md
import xdrlib
import warnings
import time
from pdbfixer import PDBFixer
from sys import stdout
import numpy as np
from sklearn.preprocessing import StandardScaler


from pdbfixer import PDBFixer
from simtk.openmm.app import PDBFile
import os


start_time = time.time()

# def fix_pdb(pdb_id):
#     path = os.getcwd()
#     if len(pdb_id) != 4:
#         print("Creating PDBFixer...")
#         fixer = PDBFixer(filename=pdb_id)
#         print("Finding missing residues...")
#         fixer.findMissingResidues()

#         chains = list(fixer.topology.chains())
#         keys = fixer.missingResidues.keys()
#         for key in list(keys):
#             chain = chains[key[0]]
#             if key[1] == 0 or key[1] == len(list(chain.residues())):
#                 print("ok")
#                 del fixer.missingResidues[key]

#         print("Finding nonstandard residues...")
#         fixer.findNonstandardResidues()
#         print("Replacing nonstandard residues...")
#         fixer.replaceNonstandardResidues()
#         print("Removing heterogens...")
#         fixer.removeHeterogens(keepWater=True)

#         print("Finding missing atoms...")
#         fixer.findMissingAtoms()
#         print("Adding missing atoms...")
#         fixer.addMissingAtoms()
#         print("Adding missing hydrogens...")
#         fixer.addMissingHydrogens(7.0)
#         print("Writing PDB file...")

#         fixed_pdb_file = os.path.join(path, "%s_fixed_pH_%s.pdb" % (pdb_id.split('.')[0], 7))
#         with open(fixed_pdb_file, "w") as outfile:
#             PDBFile.writeFile(
#                 fixer.topology, 
#                 fixer.positions, 
#                 outfile, 
#                 keepIds=True
#             )
#         return fixed_pdb_file

# fixed_pdb = fix_pdb('data/8jn1.pdb')

# if fixed_pdb:
#     print(f"Fixed PDB saved at {fixed_pdb}")
# else:
#     print("PDB fixing failed.")


from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import numpy as np

# File paths
pdb5_file = 'data/7dmu_fixed_pH_7.pdb'
force_field_files = ['amber14-all.xml', 'amber14/tip3p.xml']

# Load initial coordinates from a PDB file
pdb = PDBFile(pdb5_file)

# Choosing force field parameters
ff = ForceField(*force_field_files)
system = ff.createSystem(pdb.topology, nonbondedMethod=CutoffNonPeriodic)

# Experiment parameters
temperature = 300 * kelvin
friction_coeff = 1 / picosecond
time_step = 0.002 * picoseconds
total_steps = 10000000
save_interval = 10000  # Interval at which to save the coordinates

# Calculate how many data points will be saved
num_data_points = total_steps // save_interval

# Integrator
integrator = LangevinIntegrator(temperature, friction_coeff, time_step)

# Set a seed for Langevin integrator for reproducibility
seed = 42
integrator.setRandomNumberSeed(seed)

# Create a simulation object
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.minimizeEnergy()

# Initialize an array to store the selected data points
positions_array = np.zeros((num_data_points, pdb.topology.getNumAtoms(), 3))
# rmsd_array = np.zeros(num_data_points)
energy_array = np.zeros(num_data_points)

# Reference structure for RMSD (initial state)
# reference_structure = simulation.context.getState(getPositions=True).getPositions(asNumpy=True) / 10.0

# Run the simulation and collect data at specified intervals
data_index = 0
for step in range(1, total_steps + 1):
    simulation.step(1)
    if step % save_interval == 0:
        state = simulation.context.getState(getPositions=True, getEnergy=True)
        positions_array[data_index] = state.getPositions(asNumpy=True).value_in_unit(nanometers) / 10.0

        # Calculate RMSD with respect to the reference structure
        # rmsd_array[data_index] = rmsd(positions_array[data_index], reference_structure, superposition=True)

        # Store potential energy (in kJ/mol)
        energy_array[data_index] = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)

        data_index += 1

        # Print progress update
        ps_completed = (step * time_step)
        print(f"{ps_completed:} completed, {data_index}/{num_data_points} data points saved")
        import sys
        sys.stdout.flush()

# Initialize lists to store PDB information
residue_numbers = []
residue_names = []
atom_names = []

# Extract residue numbers, names, and atom names from the PDB file
with open(pdb5_file, 'r') as pdb_file:
    for line in pdb_file:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            columns = line.split()
            atom_names.append(columns[2])
            residue_names.append(columns[3])
            residue_numbers.append(int(columns[5]))

# Determine the number of intervals (e.g., 10000th steps included)
num_intervals = total_steps // save_interval

# Define dtype for the structured array
dtype = [
    ('residue_number', 'i4'),
    ('residue_name', 'U3'),
    ('atom_name', 'U3'),
    ('x', 'f4'),
    ('y', 'f4'),
    ('z', 'f4')
]

structured_array = np.zeros((num_intervals, pdb.topology.getNumAtoms()), dtype=dtype)

# Fill the structured array with data from the selected positions and the PDB file
for interval_index in range(num_intervals):
    for atom_index in range(pdb.topology.getNumAtoms()):
        structured_array[interval_index, atom_index] = (
            residue_numbers[atom_index],
            residue_names[atom_index],
            atom_names[atom_index],
            positions_array[interval_index, atom_index, 0],
            positions_array[interval_index, atom_index, 1],
            positions_array[interval_index, atom_index, 2]
        )

# Save the structured array with selected positions and PDB information
np.save('data/7dmu_pos.npy', structured_array)
# np.save('data/7dmu_rmsd.npy', rmsd_array)
np.save('data/7dmu_energy.npy', energy_array)

print("Dataset created")

# End measuring time
end_time = time.time()

# Calculate total time in seconds
total_time = end_time - start_time

# Print the total simulation time in seconds, minutes, and hours
print(f"Total simulation time: {total_time:.2f} seconds ({total_time/60:.2f} minutes, {total_time/3600:.2f} hours)")
