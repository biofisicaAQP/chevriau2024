
import subprocess, logging, tempfile, argparse
from natsort import natsorted
from pathlib import Path
import pandas as pd
import numpy as np
import re



def OptionsParser() -> argparse.Namespace:
    
    '''parses the command line arguments and returns the parsed arguments as a Namespace object'''

    parser = argparse.ArgumentParser(description='Z-axis Density Extract')
    parser.add_argument('-top', '--topofile', help='Topology file')
    parser.add_argument('-traj', '--trajdir', help='Trajectory folder')
    return parser.parse_args()



def run_cpptraj(topofile: str, trajdir: str, first_aox: int, last_aox: int) -> None:
    """
    Runs cpptraj to extract dipole and dihedral data for a given range of residues.

    Args:
        topofile (str): The topology file (e.g., .prmtop file).
        trajdir (str): The directory containing the trajectory files (e.g., .nc files).
        first_aox (int): The initial residue ID (e.g., 1).
        last_aox (int): The final residue ID (e.g., 10).

    Returns:
        None
    """

    # Set up logging
    logging.basicConfig(filename='residencias.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

    # Validate input
    if not Path(topofile).exists():
        raise FileNotFoundError(f'Topology file {topofile} not found.')
    if not Path(trajdir).exists():
        raise FileNotFoundError(f'Trajectory directory {trajdir} not found.')
    if first_aox >= last_aox:
        raise ValueError('First residue ID must be less than last residue ID.')

    # Create output directories
    Path('./dipole').mkdir(parents=True, exist_ok=True)
    Path('./dihedral').mkdir(parents=True, exist_ok=True)

    # Generate cpptraj input script
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write(f'parm {topofile}\n')

        trajectory_files = [file for file in natsorted(Path(trajdir).glob('*.nc'))]
        for file in trajectory_files:
            temp_file.write(f'trajin {file} 1 last 1000\n')
        temp_file.write(f'autoimage \n')

        for res in range(first_aox, last_aox):
            temp_file.write("vector dipole out ./arR_A.dat :87,216,225,231\n") # Also, extract coordinates center of ar/R for alignment
            temp_file.write(f"vector dipole out ./dipole/{res}_dipo.dat :{res}\n")
            temp_file.write(f"dihedral out ./dihedral/{res}_dihe.dat :{res}@H1 :{res}@O1 :{res}@O2 :{res}@H2   \n")

        temp_file.write(f'go\n')
        temp_file.write(f'exit\n')

    # Run cpptraj
    subprocess.run(['cpptraj', '-i', temp_file.name], stdout=subprocess.PIPE)

    # Log completion message
    logging.info(f'Finished cpptraj calculation for residues {first_aox} to {last_aox}')


def cylindrical_filter(data: np.array, radius: float) -> np.array:
    '''
    Returns a filter mask based on cylindrical distance.

    Args:
        data (np.array): The input data array with x, y, and z coordinates.
        radius (float): The radius of the cylinder.

    Returns:
        np.array: The filter mask indicating which points are within the cylindrical distance.
    '''
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    cylindrical_distance = x**2 + y**2
    filter_mask = cylindrical_distance < (radius**2)
    return filter_mask

def get_file_res_id(x: str) -> str:
    """
    Extracts the file residue ID from the given string.
    Args:
        x (str): The input string.
    Returns:
        str: The extracted residue ID.
    """
    return re.search(r"(\d+)", str(x)).group(0)



def merge_dihedral_dipole_files() -> None:
    """
    Merges dihedral and dipole files and saves the modified dipole files in a new directory.
    """
    # Create the new directory to store modified dipole files
    Path('./modified_dipole').mkdir(parents=True, exist_ok=True)

    dihedral_files = [file for file in natsorted(Path("./dihedral").glob('*_dihe.dat'))]

    for dihe_file in dihedral_files:

        res_id = get_file_res_id(dihe_file)

        dipole_file = f"dipole/{res_id}_dipo.dat"

        dipole_data = pd.read_csv(dipole_file, skiprows=1, delim_whitespace=True, names=["Frame", "dipox", "dipoy", "dipoz", "x", "y", "z"])
        dihe_data = pd.read_csv(dihe_file, skiprows=1, delim_whitespace=True, names=["frame", "dihedral"])

        # Add the dihedral column to the dipole data
        dipole_data["dihedral"] = dihe_data["dihedral"]

        # Save the modified file in the new directory
        dipole_data.to_csv(f"modified_dipole/{res_id}_dipo.dat", index=False)
        
        
def center_filter_dipole_file(dipole_file: str, arr_coords: pd.DataFrame, radius: float) -> pd.DataFrame:
    """
    Filters and centers the dipole data based on cylindrical distance.
    Args:
        dipole_file (str): The path to the dipole file.
        arr_coords (pd.DataFrame): The array of ar/R coordinates to center.
        radius (float): The radius of the cylinder.
    Returns:
        pd.DataFrame: The filtered and centered dipole data.
    """
    dipole_data = pd.read_csv(dipole_file)
    dipole_data[["x", "y", "z"]] = (dipole_data[["x", "y", "z"]].to_numpy() - arr_coords[["x", "y", "z"]].to_numpy())
    dipole_data = dipole_data[cylindrical_filter(dipole_data[["x", "y", "z"]].values, radius)]
    
    result = dipole_data[["Frame", "dipoz", "z", "dihedral"]]
    if result.empty:
        return None
    else:
        result["molecule"] = "esto"
        return result
    
def merge_filtered_files(arr_coords: pd.DataFrame, radius: float) -> pd.DataFrame:
    """
    Merges and filters the dipole data based on cylindrical distance.
    Args:
        arr_coords (pd.DataFrame): The array of ar/R coordinates to center.
        radius (float): The radius of the cylinder.
    Returns:
        pd.DataFrame: The merged and filtered dipole data.
    """
    all_results = []
    modified_dipole_files = [file for file in natsorted(Path("./modified_dipole").glob('*_dipo.dat'))]
    for file in modified_dipole_files:
        result = center_filter_dipole_file(file, arr_coords, radius)
        if result is not None:
            all_results.append(result)
    all_results = pd.concat(all_results, ignore_index=True)
    return all_results

    # Main program execution
if __name__ == '__main__':
    # Parse command line arguments
    args = OptionsParser()
    
    run_cpptraj(args.topofile, args.trajdir, 3400, 3900)

    merge_dihedral_dipole_files()
    
    arr_coords = pd.read_csv("arR_A.dat", skiprows=1, delim_whitespace=True, names=["Frame", "x", "y", "z", "a", "b", "c"])[["x", "y", "z"]]

    dipole_dihedral_data = merge_filtered_files(arr_coords, 6)

    dipole_dihedral_data.to_csv("dipole_dihedral_data.csv", index=False)