#! /usr/env python

import pandas as pd #type: ignore
import re, subprocess, tempfile, os, logging
from typing import Tuple, Optional

# Set up
res_init = 1
res_fin = 285
distance = 3.5
offset = 50
n_iterations = (res_fin - res_init + 1) // offset + ((res_fin - res_init + 1) % offset != 0)
pattern = re.compile(":([\d]+)@O1_:([\d]+)@([A-Za-z0-9]+)")


def extract_resid(column_name: str) -> Optional[Tuple[int, int]]:
    """
    Extracts the residue IDs from a column name.

    Args:
        column_name (str): The column name.

    Returns:
        Optional[Tuple[int, int]]: A tuple containing the residue IDs if found, else None.
    """
    match = pattern.search(column_name)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    else:
        return None
    
    
    
def run_cpptraj(new_res_init: int, new_res_fin: int, distance: float) -> None:
    """
    Runs cpptraj to calculate contacts for a range of residues.

    Args:
        new_res_init (int): The initial residue ID.
        new_res_fin (int): The final residue ID.
        distance (float): The distance threshold for native contacts.

    Returns:
        None
    """
    # Set up logging
    logging.basicConfig(filename='residencias.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
    
    # Create a temporary file to store cpptraj input
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write(f'parm *.parm7\n')
        temp_file.write(f'trajin Trayectoria/[0-9]00ns.nc 1 last \n')
        temp_file.write(f'trajin Trayectoria/[1-2][0-9]000ns.nc  1 last \n')
        temp_file.write(f'autoimage \n')
        temp_file.write(f'nativecontacts :AOX&@O1 :{new_res_init}-{new_res_fin}&!@H= distance {distance} byresidue series skipnative seriesnnout {new_res_init}-{new_res_fin}.dat\n')
        temp_file.write(f'go\n')
        temp_file.write(f'exit\n')
    
    # Run cpptraj command
    subprocess.run(['cpptraj', '-i', temp_file.name], stdout=subprocess.PIPE)
    
    # Log completion message
    logging.info(f'Finished cpptraj calculation for residues {new_res_init} to {new_res_fin}.')
    
    


def group_by_atoms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups the DataFrame columns by atoms IDs.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The grouped DataFrame.
    """
    groups = {}
    for column_name in df.columns:
        resid = extract_resid(column_name)
        if resid:
            if resid in groups:
                groups[resid].append(column_name)
            else:
                groups[resid] = [column_name]

    grouped_dfs = []

    for group in groups.values():
        grouped_df = df[group]
        grouped_df = grouped_df.any(axis=1)
        grouped_dfs.append(grouped_df)

    grouped_df = pd.concat(grouped_dfs, axis=1)

    new_columns = [f"{resAOX}_{resProt}" for resAOX, resProt in groups.keys()]
    grouped_df.columns = new_columns

    return grouped_df



def group_by_residues(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups the DataFrame columns by residue IDs.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The grouped DataFrame.
    """
    groups = {}

    # Iterate over the columns of the DataFrame
    for i, resid in enumerate([int(column.split("_")[1]) for column in df.columns if "_" in column]):
        column_name = df.columns[i]
        # Check if the residue ID is already in the groups dictionary
        if resid in groups:
            groups[resid].append(column_name)
        else:
            groups[resid] = [column_name]

    grouped_dfs = []

    # Iterate over the groups and create a DataFrame for each group
    for group in groups.values():
        grouped_df = df[group]
        grouped_df = grouped_df.any(axis=1).astype(int)
        grouped_dfs.append(grouped_df)

    # Concatenate the grouped DataFrames into a single DataFrame
    grouped_df = pd.concat(grouped_dfs, axis=1)

    # Create new column names for the grouped DataFrame
    new_columns = [f"residProt_{resid}" for resid in groups.keys()]
    grouped_df.columns = new_columns

    return grouped_df

def save_residues(df: pd.DataFrame, new_res_init: int, new_res_fin: int) -> None:
    """
    Saves the splitted residues lifetimes to a TSV file.
    Args:
        df (pd.DataFrame): The input DataFrame.
        new_res_init (int): The initial residue ID.
        new_res_fin (int): The final residue ID.
    Returns:
        None
    """
    total_rows = len(df)
    sums = df.sum()
    proportions = sums / total_rows
    proportions.to_csv(f"{new_res_init}-{new_res_fin}_lifetimes.tsv", sep='\t')
    
    
    
def merge_tsv_files(folder_path: str, output_file: str) -> None:
    """
    Merges multiple TSV files into a single file.

    Args:
        folder_path (str): The path to the folder containing the TSV files.
        output_file (str): The name of the output file.

    Returns:
        None
    """
    # Get the list of all TSV files in the folder
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.tsv')]
    
    # Read each file and concatenate them into a single DataFrame
    dfs = []
    for file in file_list:
        df = pd.read_csv(os.path.join(folder_path, file), sep='\t', header=None)
        dfs.append(df)
    merged_df = pd.concat(dfs)
    merged_df = merged_df.sort_values(by=1, ascending=False)

    # Remove the headers and save the DataFrame to an output file
    merged_df.to_csv(output_file, sep='\t', header=None, index=None)
    
    
def main():
    for i in range(n_iterations):
        new_res_init = res_init + i * offset
        new_res_fin = min(res_fin, new_res_init + offset - 1)

        run_cpptraj(new_res_init, new_res_fin, distance)
        
        df = pd.read_table(f"{new_res_init}-{new_res_fin}.dat", delim_whitespace=True)
        print(f"loading table {new_res_init}-{new_res_fin} ready...")
        
        grouped_by_atoms_df = group_by_atoms(df)
        print("Grouping atoms ready")

        grouped_by_residues_df = group_by_residues(grouped_by_atoms_df)
        print("Grouping ProtResid Ready")

        save_residues(grouped_by_residues_df, new_res_init, new_res_fin)
        
    merge_tsv_files(os.getcwd() , 'lifetimes.tsv')


if __name__ == '__main__':
    main()
