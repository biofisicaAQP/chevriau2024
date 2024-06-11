from lib.classes import *
from lib.functions import *
import matplotlib.pyplot as plt
import pandas as pd

def pf_calculator(filename: str, chain_id: int, ref_z_1_order: int, ref_z_2_order: int, 
                  ref_xy_1_order: int, ref_xy_2_order: int, timestep: float, pore_radius: int = 6):
    """
    Calculates the permeability factor (PF) for a specific chain in a molecular dynamics simulation.

    Args:
        filename (str): The input filename for the simulation data.
        chain_id (int): The ID of the chain to analyze.
        ref_z_1_order (int): The order of the first z reference atom.
        ref_z_2_order (int): The order of the second z reference atom.
        ref_xy_1_order (int): The order of the first xy reference atom.
        ref_xy_2_order (int): The order of the second xy reference atom.
        timestep (float): The timestep of the simulation.
        pore_radius (int, optional): The radius of the cylindrical pore. Defaults to 6.

    Returns:
        tuple: A tuple containing the calculated pf, the MSD plot, and the list of atoms.
    """

    n_ref_atoms = 3
    first_non_ref_atom = 12
    drop_msd_points = 10
    vol_h2o = 2.99003322259e-23
    ref_z_1_atom = ref_z_1_order + chain_id*n_ref_atoms
    ref_z_2_atom = ref_z_2_order + chain_id*n_ref_atoms
    ref_xy_1_atom = ref_xy_1_order + chain_id*n_ref_atoms
    ref_xy_2_atom = ref_xy_2_order + chain_id*n_ref_atoms

    data = open_dataset(filename)
    atoms_coordinates = data['coordinates']
    total_atoms = len(data.dimensions['atom'])
    total_frames = len(data.dimensions['frame'])
    top_atom, low_atom = get_z_top_low(atoms_coordinates, ref_z_1_atom, ref_z_2_atom)
    
    Pore = pore_traject(atoms_coordinates, top_atom, low_atom, ref_xy_1_atom, ref_xy_2_atom, pore_radius)
    atom_list = list(range(first_non_ref_atom,total_atoms))
    compendio_atomos = atoms_inside_pore(atoms_coordinates, atom_list, Pore)
    coord_atoms_in_pore = atoms_coordinates[:,compendio_atomos,:]
    dz = coord_atoms_in_pore[1:,:,2] - coord_atoms_in_pore[:-1,:,2]
    #dz = np.insert(dz, dz.shape[0],np.zeros(shape = dz.shape[1]), axis = 0)
    dz = np.insert(dz, 0, np.zeros(shape = dz.shape[1]), axis = 0)
    dz_exp = np.expand_dims(dz, axis=2)
    atoms_dz_array = np.concatenate(
        (coord_atoms_in_pore,dz_exp),
        axis=2)
    atoms_dz_array = drop_dz(atoms_dz_array, Pore)
    n_array = compute_n(atoms_dz_array, Pore)
#    df_n_array = pd.DataFrame(n_array)
#    df_n_array.to_csv('n_vs_time'+str(chain_id)+'.csv', sep =' ')
    fragments = 200
    n_msd = compute_msd(n_array, fragments)
    
    
    time_axis = np.arange(n_msd[drop_msd_points:].size)
    pf = (regresion_lineal(time_axis, n_msd[drop_msd_points:]))*vol_h2o/(2*timestep)
    #print(f'{pf/1e-14:.4f}e-14')
    msd_plot = plt.plot(time_axis, n_msd[drop_msd_points:], 'bo')
    return (pf, msd_plot, compendio_atomos)


