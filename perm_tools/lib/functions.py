import netCDF4 as nc
import numpy as np
from lib import classes
from sklearn import linear_model

def as_list(x) -> list:
    """
    Converts input to a list if it is not already a list.

    Args:
        x: The input to be converted to a list.

    Returns:
        list: The input converted to a list, or the input itself if it is already a list.
    """
    res = [x]
    if type(x) is list:
        res = x
    return res

def open_dataset(filename: str) -> nc.Dataset:
    """
    Opens a NetCDF dataset file.

    Args:
        filename (str): The path to the NetCDF dataset file.

    Returns:
        netCDF4.Dataset: The opened NetCDF dataset.
    """

    return nc.Dataset(filename)


def import_coords(data: np.ndarray, start: int, stop: int = None) -> np.ndarray:
    """
    Imports coordinates from a 3D array.

    Args:
        data (numpy.ndarray): The 3D array containing the coordinates.
        start (int): The starting index for slicing the array.
        stop (int, optional): The stopping index for slicing the array. Defaults to None.

    Returns:
        numpy.ndarray: The sliced 3D array based on the start and stop indices.
    """
    return data[start:,:,:] if stop is None else data[start:stop,:,:]

def get_z_top_low(atoms_coordinates: np.ndarray, atom_1: int, atom_2: int) -> tuple:
    """
    Determines the top and low atoms based on their average z-coordinate.

    Args:
        atoms_coordinates (numpy.ndarray): The 3D array of atom coordinates.
        atom_1 (int): Index of the first atom.
        atom_2 (int): Index of the second atom.

    Returns:
        tuple: A tuple containing the index of the top atom and the index of the low atom.
    """

    avg_atom1 = atoms_coordinates[:,atom_1,2].mean()
    avg_atom2 = atoms_coordinates[:,atom_2,2].mean()
    if avg_atom1 > avg_atom2:
        top_atom = atom_1
        low_atom = atom_2        
    else:
        top_atom = atom_2
        low_atom = atom_1
    return(top_atom, low_atom)



def pore_traject(atoms_coordinates: np.ndarray, top_atom: int, low_atom: int,
                ref_xy_1_atom: int, ref_xy_2_atom: int, pore_radius: float) -> classes.Pore_cylinder_traj:
    """
    Generates a trajectory for a pore cylinder based on atom coordinates and reference atoms.

    Args:
        atoms_coordinates (numpy.ndarray): The 3D array of atom coordinates.
        top_atom (int): Index of the top atom.
        low_atom (int): Index of the low atom.
        ref_xy_1_atom (int): Index of the first reference atom for xy coordinates.
        ref_xy_2_atom (int): Index of the second reference atom for xy coordinates.
        pore_radius (float): The radius of the pore cylinder.

    Returns:
        classes.Pore_cylinder_traj: An instance of Pore_cylinder_traj representing the trajectory of the pore cylinder.
    """

    # Coordenadas de los átomos de referencia (3)
    a_top = np.mean(atoms_coordinates[:,as_list(top_atom),2], axis=1)
    a_low = np.mean(atoms_coordinates[:,as_list(low_atom),2], axis=1)
    a1_xy = np.mean(atoms_coordinates[:,as_list(ref_xy_1_atom),:2], axis=1)
    a2_xy = np.mean(atoms_coordinates[:,as_list(ref_xy_2_atom),:2], axis=1)
    return classes.Pore_cylinder_traj(a_top,a_low,a1_xy,a2_xy,pore_radius)


def regresion_lineal(columna_x: np.ndarray, columna_y: np.ndarray) -> float:
    """
    Performs linear regression on the given x and y data columns.

    Args:
        columna_x (numpy.ndarray): The x data column for regression.
        columna_y (numpy.ndarray): The y data column for regression.

    Returns:
        float: The coefficient of the linear regression model.
    """
    ajuste = linear_model.LinearRegression()
    ajuste.fit(columna_x.reshape((-1,1)),columna_y)
    return ajuste.coef_[0]


def compute_msd(n_array: np.ndarray, fragments: int) -> np.ndarray:
    """
    Computes the mean squared displacement (MSD) of an array split into fragments.

    Args:
        n_array (numpy.ndarray): The input array to compute MSD on.
        fragments (int): The number of fragments to split the array into.

    Returns:
        numpy.ndarray: The mean squared displacement values calculated for each fragment.
    """

    splitted_array = np.array(np.split(n_array, fragments))
    split_to_zero = (splitted_array.T-splitted_array[:,0]).T
    return np.mean(split_to_zero**2, axis = 0)


def compute_n(atoms_dz_array: np.ndarray, pore: classes.Pore_cylinder_traj) -> np.ndarray:
    """
    Computes the normalized cumulative sum of a given array based on pore length.

    Args:
        atoms_dz_array (numpy.ndarray): The array of atom dz values.
        pore (Pore): The pore object containing length information.

    Returns:
        numpy.ndarray: The normalized cumulative sum array computed based on pore length.
    """

    dn_array = (atoms_dz_array[:,:,3].sum(axis = 1))/pore.length
    return np.cumsum(dn_array)

def filter_by_radius(atoms_coordinates: np.ndarray, Pore: classes.Pore) -> np.ndarray:
    """
    Filters atom coordinates based on their distance from the center of a pore.

    Args:
        atoms_coordinates (numpy.ndarray): The array of atom coordinates.
        Pore (Pore): The Pore object containing center and radius information.

    Returns:
        numpy.ndarray: The filtered array of atom z-coordinates based on the distance from the pore center.
    """

    resta = np.subtract(atoms_coordinates[:,:,0:2],np.expand_dims(Pore.xy_center, axis=1))
    distance_to_center = np.power(((np.power(resta,2)).sum(axis = 2)),0.5)
    #distance_to_center = (((atoms_coordinates[:,:,0:2] - np.expand_dims(Pore.xy_center, axis=1))**2).sum(axis = 2))**(1/2)
    condition_1 = distance_to_center <= Pore.radius
    return (atoms_coordinates[:,:,2]-np.expand_dims(Pore.low,axis=1))[condition_1]

def filter_by_z(atoms_coordinates: np.ndarray, top: float, bot: float) -> np.ndarray:
    """
    Filters atom coordinates based on their z-coordinate values.

    Args:
        atoms_coordinates (numpy.ndarray): The array of atom z-coordinates.
        top (float): The upper bound z-coordinate for filtering.
        bot (float): The lower bound z-coordinate for filtering.

    Returns:
        numpy.ndarray: The filtered array of atom z-coordinates within the specified range.
    """

    condition_1 = atoms_coordinates <= top
    condition_2 = atoms_coordinates >= bot
    return atoms_coordinates[condition_1 & condition_2]

def compute_histogram(array: np.ndarray, bins: int or np.ndarray) -> tuple: # type: ignore
    """
    Computes the histogram of an array with specified bins.

    Args:
        array (numpy.ndarray): The input array for computing the histogram.
        bins (int or sequence of scalars): The number of bins or bin edges for the histogram.

    Returns:
        tuple: A tuple containing the histogram values and the bin edges.
    """

    arr_hist, edges = np.histogram(array, bins=bins)
    return (arr_hist, edges)


def compute_density(arr_hist: np.ndarray, edges: np.ndarray, array_shape: int) -> tuple:
    """
    Computes the density values based on histogram data and array shape.

    Args:
        arr_hist (numpy.ndarray): The histogram values.
        edges (numpy.ndarray): The bin edges of the histogram.
        array_shape (int): The shape of the original array.

    Returns:
        tuple: A tuple containing the density values and the adjusted bin edges.
    """

    density = arr_hist/(array_shape*np.diff(edges))
    return density, edges[:-1]

def drop_dz(atoms_dz_array: np.ndarray, pore: classes.Pore) -> np.ndarray:
    """
    Drops the dz values of atoms outside the specified pore boundaries.

    Args:
        atoms_dz_array (numpy.ndarray): The array of atom dz values.
        pore (Pore): The Pore object defining the pore boundaries.

    Returns:
        numpy.ndarray: A copy of the input array with dz values set to 0 for atoms outside the pore boundaries.
    """

    copy_dz_array = atoms_dz_array.copy()
    distance_to_center = (((atoms_dz_array[:,:,0:2] - np.expand_dims(pore.xy_center, axis=1))**2).sum(axis = 2))**(1/2)
    condition_1 = ((pore.low >= atoms_dz_array[:,:,2].T) | (atoms_dz_array[:,:,2].T >= pore.top)).T
    condition_2 = (distance_to_center >= pore.radius)
    copy_dz_array[1:,:,3][((condition_1[1:] | condition_2[1:]) & (condition_1[:-1] | condition_2[:-1]))] = 0.0
    return copy_dz_array


def hacer_comparacion_un_solo_saque(atoms_coordinates: np.ndarray, frame: int, atom_list: list, Pore: classes.Pore) -> list:
    """
    Compares atom coordinates with pore boundaries for a single frame extraction.

    Args:
        atoms_coordinates (numpy.ndarray): The array of atom coordinates.
        frame (int): The frame index for extraction.
        atom_list (list): The list of atom indices to compare.
        Pore (Pore): The Pore object containing boundary information for the frame.

    Returns:
        list: A list of atom indices that fall within the specified pore boundaries for the given frame.
    """

    pore_cylinder = Pore[frame]
    atom_coord = atoms_coordinates[frame,atom_list,:]
    distance_to_center = (((atom_coord[:,0:2] - pore_cylinder.xy_center)**2).sum(axis = 1))**(1/2)
    condition_1 = distance_to_center < pore_cylinder.radius
    condition_2 = (pore_cylinder.low.min() < atom_coord[:,2]) & (atom_coord[:,2] < pore_cylinder.top.max())
    true_list = np.where((condition_1 & condition_2) == True)[0].tolist()
    return [atom_list[i] for i in true_list]


def is_in_cylinder(atoms_coordinates: np.ndarray, frame: int, atom: int, Pore: classes.Pore) -> bool:
    """
    Checks if a single atom is inside a cylindrical pore for a specific frame.

    Args:
        atoms_coordinates (numpy.ndarray): The array of atom coordinates.
        frame (int): The frame index for extraction.
        atom (int): The index of the atom to check.
        Pore (Pore): The Pore object containing boundary information for the frame.

    Returns:
        bool: True if the atom is inside the cylindrical pore, False otherwise.
    """
    pore_cylinder = Pore[frame]
    atom_coord = atoms_coordinates[frame,atom,:]
    distance_to_center = ((atom_coord[:2] - pore_cylinder.xy_center) ** 2).sum(
        axis=0
    ) ** (1 / 2)
    condition_1 = distance_to_center < pore_cylinder.radius
    condition_2 = (pore_cylinder.low < atom_coord[2]) & (atom_coord[2] < pore_cylinder.top)
    return (condition_1 & condition_2)

def ingreso_egreso_lateral(atoms_coordinates: np.ndarray, frame: int, atom: int, Pore: classes.Pore, ingreso: bool = True) -> bool:
    """
    Checks if a single atom is entering or exiting laterally from a cylindrical pore for a specific frame.

    Args:
        atoms_coordinates (numpy.ndarray): The array of atom coordinates.
        frame (int): The frame index for extraction.
        atom (int): The index of the atom to check.
        Pore (Pore): The Pore object containing boundary information for the frame.
        ingreso (bool, optional): Flag indicating if it is an entry (True) or exit (False). Defaults to True.

    Returns:
        bool: True if the atom is entering or exiting laterally from the cylindrical pore, False otherwise.
    """

    if ingreso:
        frame -= 1
    pore_cylinder = Pore[frame]
    atom_coord = atoms_coordinates[frame,atom,:]
    distance_to_center = ((atom_coord[:2] - pore_cylinder.xy_center) ** 2).sum(
        axis=0
    ) ** (1 / 2)
    condition_1 = distance_to_center > pore_cylinder.radius
    condition_2 = (pore_cylinder.low < atom_coord[2]) & (atom_coord[2] < pore_cylinder.top)
    return(condition_1 & condition_2)

def dist_interatom(atom_coordinates: np.ndarray, frame: int, atom_1: int, atom_2: int) -> float:
    """
    Calculates the Euclidean distance between two atoms in a specific frame.

    Args:
        atom_coordinates (numpy.ndarray): The array of atom coordinates.
        frame (int): The frame index for extraction.
        atom_1 (int): The index of the first atom.
        atom_2 (int): The index of the second atom.

    Returns:
        float: The Euclidean distance between the two specified atoms in the given frame.
    """

    coord_atom_1 = atom_coordinates[frame, atom_1,:]
    coord_atom_2 = atom_coordinates[frame, atom_2,:]
    return np.sqrt(
        (coord_atom_1[0] - coord_atom_2[0]) ** 2
        + (coord_atom_1[1] - coord_atom_2[1]) ** 2
        + (coord_atom_1[2] - coord_atom_2[2]) ** 2
    )


def por_donde_pasa(atoms_coordinates: np.ndarray, frame: int, top_atom: int, low_atom: int, atom: int) -> int:
    """
    Determines the path an atom takes between two reference atoms in a specific frame.

    Args:
        atoms_coordinates (numpy.ndarray): The array of atom coordinates.
        frame (int): The frame index for extraction.
        top_atom (int): The index of the top reference atom.
        low_atom (int): The index of the low reference atom.
        atom (int): The index of the atom to determine the path.

    Returns:
        int: The index of the reference atom (low_atom or top_atom) that the atom is closer to in the given frame.
    """
    dist_to_low = dist_interatom(atoms_coordinates, frame, low_atom, atom)
    dist_to_top = dist_interatom(atoms_coordinates, frame, top_atom, atom)
    return low_atom if dist_to_low < dist_to_top else top_atom
 

def atoms_inside_pore(atoms_coordinates: np.ndarray, atom_list: list, Pore: classes.Pore) -> list:
    """
    Identifies atoms that are inside a cylindrical pore across multiple frames.

    Args:
        atoms_coordinates (numpy.ndarray): The array of atom coordinates for all frames.
        atom_list (list): The list of atom indices to check.
        Pore (Pore): The Pore object defining the cylindrical pore.

    Returns:
        list: A list of unique atom indices that are inside the cylindrical pore in any frame.
    """

    compendio_atomos = []
    for n_frame, frame in enumerate(atoms_coordinates):
        if n_frame % 500 == 0:
            print(n_frame)
        lista_true_atomos = hacer_comparacion_un_solo_saque(atoms_coordinates, n_frame, atom_list, Pore)
        compendio_atomos.extend(lista_true_atomos)
    return list(set(compendio_atomos))
