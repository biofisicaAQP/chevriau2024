#! /usr/bin/env python

import pf_calculator.pf_calculator as pf_calc
import os
import matplotlib.pyplot as plt
import pandas as pd
import json
import argparse
import itertools

def argument_parser():
    """Parses command-line arguments.

    Returns:
        argparse.Namespace: An object containing parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Calculates the permeability of a given molecule')
    parser.add_argument('-m', "--mol", type=str, default="molecule", help='Molecule name')
    parser.add_argument('-d', "--dir", type=str, default=".", help='Folder where the pf simulation files are located')
    return parser.parse_args()


def calculate_permeability(filename, chain_id, ref_z_1, ref_z_2, ref_xy_1, ref_xy_2, timestep, chains_letter, args, pf_dict_list):
    """Calculates the permeability for a given chain and saves the results.

    Args:
        filename (str): Path to the simulation file.
        chain_id (int): Index of the chain.
        ref_z_1 (float): First reference z-coordinate.
        ref_z_2 (float): Second reference z-coordinate.
        ref_xy_1 (float): First reference xy-coordinate.
        ref_xy_2 (float): Second reference xy-coordinate.
        timestep (float): Simulation timestep.
        chains_letter (list): List of chain labels.
        args (argparse.Namespace): Parsed arguments.
        pf_dict_list (list): List of dictionaries containing permeability data.
    """
    (pf, plot, compendio_atomos) = pf_calc.pf_calculator(filename, chain_id, ref_z_1, ref_z_2, ref_xy_1, ref_xy_2, timestep=timestep)
    
    # Save MSD plot
    plt.xlabel('Time (ps)')
    plt.ylabel('MSD')
    plt.savefig(f'{os.path.splitext(filename)[0]}_atomsinpore{chains_letter[chain_id]}_msd.png')
    plt.close()

    # Print permeability and save to file
    print(f'{os.path.dirname(filename)}_{chains_letter[chain_id]}_pf: {pf/1e-14:.4f}e-14')
    pf_dict = {'Time': os.path.basename(filename), 'Molecule': args.mol, 'Chain': chains_letter[chain_id], 'pf': pf}
    with open(f"{filename}_pf.txt", "a") as myfile:
        myfile.write(json.dumps(pf_dict))
    pf_dict_list.append(pf_dict)

    # Save list of atoms in the pore
    fname_atom_list = f'{os.path.splitext(filename)[0]}_atomsinpore{chains_letter[chain_id]}.txt'
    with open(fname_atom_list, 'w') as f:
        for atom in compendio_atomos:
            f.write("%s\n" % atom)

if __name__ == "__main__":
    args = argument_parser()

    # Get all nc files in the specified directory and its subdirectories
    files = [os.path.join(r, file) for r, d, f in os.walk(args.dir) for file in f if file.endswith('.nc')]

    # Get unique directories from the file list
    dirs = list({os.path.dirname(file) for file in files})

    chains = [0, 1, 2, 3]
    chains_letter = ['A', 'B', 'C', 'D']

    timestep = 10**(-12)
    ref_z_1, ref_z_2, ref_xy_1, ref_xy_2 = 0, 1, 1, 2

    pf_dict_list = []

    for filename, chain_id in itertools.product(files, chains):
        calculate_permeability(filename, chain_id, ref_z_1, ref_z_2, ref_xy_1, ref_xy_2, timestep, chains_letter, args, pf_dict_list)

    # Create DataFrame from permeability data and save to CSV
    pf_dataframe = pd.DataFrame(pf_dict_list)
    print(pf_dataframe)
    pf_dataframe.to_csv('pf_dataframe.csv')
