#!/usr/bin/env python
import sys
import perm_event.perm_event as perm_ev_calc
import pandas as pd

def process_file(filename: str, chains: list, chains_letter: list, 
                ref_z_1: int, ref_z_2: int, ref_xy_1: int, ref_xy_2: int) -> None:
    """
    Processes the specified file to calculate permeation events for each chain and saves the results to a CSV file.

    Args:
        filename (str): The input filename to process.
        chains (list): The list of chain IDs to process.
        chains_letter (list): The corresponding chain letters.
        ref_z_1 (int): Index for the first z reference.
        ref_z_2 (int): Index for the second z reference.
        ref_xy_1 (int): Index for the first xy reference.
        ref_xy_2 (int): Index for the second xy reference.

    Returns:
        None
    """
    list_events_total = []
    for chain_id in chains:
        (list_events_chain, n_eventos) = perm_ev_calc.perm_event_calculator(filename, chain_id, ref_z_1, ref_z_2, ref_xy_1, ref_xy_2, chains_letter)
        print(f'Chain {chains_letter[chain_id]} perm events: {n_eventos}')
        list_events_total += list_events_chain

    event_df = pd.DataFrame(list_events_total)
    event_df.to_csv('event_df.csv', index=False)


def main():
    
    if len(sys.argv) < 2:
        #use the input file from the command line, NetCDF file of the trajectory, reference CA and water/H2O2 molecules only
        print("Please provide the input filename as an argument.")
        sys.exit(1)

    filename = sys.argv[1]
    chains = [0, 1, 2, 3]
    chains_letter = ['A', 'B', 'C', 'D']
    (ref_z_1, ref_z_2, ref_xy_1, ref_xy_2) = (0, 2, 1, 2)

    process_file(filename, chains, chains_letter, ref_z_1, ref_z_2, ref_xy_1, ref_xy_2)


if __name__ == "__main__":
    main()
