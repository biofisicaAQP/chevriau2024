#! /usr/bin/env -S python

import pandas as pd
import re
import os
import natsort
import argparse
import numpy as np
from typing import List

def option_parser() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Script options')
    parser.add_argument('-f', '--file', type=str, help='Input file')
    parser.add_argument('-o', '--output', type=str, default="output",help='Output file')
    parser.add_argument('-t', '--type', type=str, help='a (protein residues as aceptor) or d (protein residues as donor)', required=True)
    parser.add_argument('-c', '--chunk', type=int, default=1000, help='Chunk size')
    return parser.parse_args()


def get_columns(input_file: str, type_a: str) -> List[str]:
    """
    Get the column names based on the input file and type.
    Args:
        input_file (str): Path to the input file.
        type_a (str): Type of column names ('a' or 'd').
    Returns:
        List[str]: List of column names.
    """
    data = pd.read_csv(input_file, nrows=0, delim_whitespace=True)
    if type_a == "a":
        column_names = [col.split('@')[0] for col in data.columns]
    elif type_a == "d":
        column_names = [col.split('-')[1].split('@')[0] for col in data.columns if '-' in col]
        column_names.insert(0, data.columns[0].split('@')[0])
    else:
        raise ValueError("Invalid value. Must be 'a' or 'd'.")
    return column_names


def process_chunks(input_file: str, output_file: str, chunk_size: int, column_names: List[str]) -> None:
    """
    Process the data in chunks, since the file is too big to load and calculate in memory, save temporary files.
    
    Args:
        input_file (str): Path to the input file.
        output_file (str): Path to the output file.
        chunk_size (int): Size of each chunk.
        column_names (List[str]): List of column names.
    """
    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size, delim_whitespace=True)):
        result = {}
        print(f"Processing Chunk {i}")
        chunk.columns = column_names
        unique_columns = np.unique(column_names)
        
        for column in unique_columns:
            if column not in result:
                result[column] = []
            if column == "#Frame":
                result[column].extend(list(chunk[column]))
            if column != "#Frame":
                # 1 if any atom (column) in the residue has Hbond, 0 otherwise
                array = chunk.filter(like=column).to_numpy()
                suma = np.sum(array, axis=1)
                result[column].extend(list(np.where(suma >= 1, 1, 0)))
        df = pd.DataFrame(result)
        df.to_csv(f"{output_file}_{i}", index=False)


def get_files(output_file: str) -> List[str]:
    """
    Get a list of files that match the temporary file pattern.

    Args:
        output_file (str): file pattern.

    Returns:
        List[str]: A list of matching file names.
    """
    regex = re.compile(output_file + r'_[0-9]{1,3}')
    files = [f for f in os.listdir('.') if re.match(regex, f)]
    return natsort.natsorted(files)




def process_files(files: List[str], output_file: str) -> None:
    """
    Process the temporary files and combine the data into a single dataframe
    Calculate fraction of all frames that have a hydrogen bond for each residue.
    
    Args:
        files (List[str]): List of temporary file names.
        output_file (str): Path to the output file.
    """
    dataframes = []
    for file in files:
        df = pd.read_csv(file)
        dataframes.append(df)

    combined_df = pd.concat(dataframes)

    result_dict = {
        key: (
            round(len(value), 1)
            if key == '#Frame'
            else round((value.sum() / len(value)), 3)
        )
        for key, value in combined_df.items()
    }

    data = pd.DataFrame(result_dict, index=[0])
    data = data.transpose()

    data.to_csv(f"{output_file}.csv", sep=';', decimal=',')


def main():
    args = option_parser()

    column_names = get_columns(args.file, args.type)
    process_chunks(args.file, args.output, args.chunk, column_names)
    files = get_files(args.output)
    process_files(files, args.output)

if __name__ == "__main__":
    main()
