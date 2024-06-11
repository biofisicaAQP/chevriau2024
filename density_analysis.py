#! /usr/bin/env -S python -u
import pytraj as pt  # type: ignore
import numpy as np # type: ignore
import os, re, time, argparse
from natsort import natsorted # type: ignore
from collections import OrderedDict
import pickle


def OptionsParser() -> argparse.Namespace:
    
    '''parses the command line arguments and returns the parsed arguments as a Namespace object'''

    parser = argparse.ArgumentParser(description='Z-axis Density Extract')
    parser.add_argument('-top', '--topofile', help='Topology file')
    parser.add_argument('-traj', '--trajdir', help='Trajectory folder')
    parser.add_argument('-type', '--type', default="AOX", help='WAT or AOX')
    parser.add_argument('-temp', '--tempdir',default = "./tmp", help='Temporary files folder')
    parser.add_argument('-start', '--start', default = "Block_1" , help='Starting Block')
    parser.add_argument('-slice', '--slice', default = 1 , help='offset of trayectory frames to analyze')
    parser.add_argument('-size', '--size', default = 5000 , help='Size of blocks to analyze')
    return parser.parse_args()


def extract_from_topology(topology: str, mask: str, block_size: int, pytraj_type: str = "vector center"):
    """
    Extracts data from topology file and formats it into blocks of residues.
    Block size is defined by the block_size parameter, depends of RAM available.

    Args:
        topology (str): The topology file.
        mask (str): The mask to apply to the topology.
        pytraj_type (str, optional): The type of pytraj. Defaults to "vector center".
        block_size (int, optional): The size of each block of residues.

    Returns:
        Dict[str, List[str]]: A dictionary where each key is a block name and each value is a list of pytraj formatted residues.
    """

    topo = pt.load_topology(topology)
    data = list(topo[mask].residues)
    
    # Extract list of residues and split into blocks
    residue_list = [residue.original_resid for residue in data]
    residue_blocks = [residue_list[i:i+block_size] for i in range(0, len(residue_list), block_size)]

    # Initialize an ordered dictionary to store the formatted data
    formatted_data = OrderedDict()

    # Define the header with arR residues, which is the same for all blocks
    header = [f"{pytraj_type} ArR_A :49,122,175,184,190",
              f"{pytraj_type} ArR_B :287,360,413,422,428",
              f"{pytraj_type} ArR_C :525,598,651,660,666",
              f"{pytraj_type} ArR_D :763,836,889,898,904",]

    # Iterate over each block and add residues to the dictionary
    for i, block in enumerate(residue_blocks):
        for residue in block:
            formatted_residues = header
            formatted_residues.append(f"{pytraj_type} RES_{str(residue)} :{str(residue).strip()}")
            formatted_data[f"Block_{i+1}"] = formatted_residues

    # Return the formatted data
    return formatted_data

class TimeControl:
    """
    A class for measuring elapsed time.

    Attributes:
        _start_time (float): The start time of the timer.

    Methods:
        start(): Start the timer.
        stop(): Stop the timer and return the elapsed time in minutes.
    """

    def __init__(self):
        self._start_time = None

    def start(self):
        """
        Start the timer by setting the start time.
        """
        self._start_time = time.time()

    def stop(self):
        """
        Stop the timer and return the elapsed time in minutes.

        Returns:
            float: The elapsed time in minutes.
        
        Raises:
            RuntimeError: If the start() method was not called before stop() method.
        """
        if self._start_time is None:
            raise RuntimeError("Debugging error: start() method was not called before stop() method.")
        final_time = time.time()
        elapsed_time = (final_time - self._start_time) / 60
        self._start_time = None
        return elapsed_time


class DataFilter:
    """
    A class that provides methods for filtering data based on cylindrical distance and arR key.

    Args:
        radius (float, optional): The radius used for cylindrical filtering. Defaults to 4.
        arR_key (str, optional): The key used for arR filtering. Defaults to "ArR_A".
    """

    def __init__(self, radius=4, arR_key="ArR_A"):
        self.radius = radius
        self.arR_key = arR_key

    def cylindrical_filter(self, data):
        """
        Filters the data based on cylindrical distance.

        Args:
            data (numpy.ndarray): The input data array.

        Returns:
            numpy.ndarray: The filtered data array.
        """
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        cylindrical_distance = x**2 + y**2
        return data[cylindrical_distance < (self.radius**2 + self.radius**2)]

    def arR_filter(self, data):
        """
        Filters the data based on the arR key.

        Args:
            data (dict): The input data dictionary.

        Returns:
            dict: The filtered data dictionary.
        """
        for key in data.keys():
            if key != self.arR_key:
                data[key] = np.array(data[key]) - np.array(data[self.arR_key])
            else:
                data[key] = np.array(data[key])
        return data

    def process_data_with_filters(self, traj_data):
        """
        Processes the trajectory data with the defined filters.

        Args:
            traj_data (dict): The trajectory data dictionary.

        Returns:
            dict: The processed data dictionary.
        """
        data_ref_chain = self.arR_filter(traj_data)
        final_data_z = {
            key: self.cylindrical_filter(data_ref_chain[key])[:, 2]
            for key in data_ref_chain.keys()
        }
        return {key: value for key, value in final_data_z.items() if value.size > 0}


def load_file_data(filename: str):
    """
    Load data from a file using pickle.

    Parameters:
    filename (str): The path to the file to be loaded.

    Returns:
    file_data: The data loaded from the file.
    """
    with open(filename, 'rb') as file:
        file_data = pickle.load(file)
        print(f"loaded {filename}")
    return file_data


def save_data_to_pickle(data: any, filename: str):
    """
    Save data to a pickle file.

    Parameters:
    data (any): The data to be saved.
    filename (str): The name of the pickle file.

    Returns:
    None
    """
    if data is None:
        return
    with open(filename, 'wb') as pickle_file:
        pickle.dump(data, pickle_file)


def combine_data(filtered_data: dict) -> dict:
    """
    Combines the filtered data into a single dictionary.
    
    Args:
        filtered_data (dict): A dictionary containing filtered data.
        
    Returns:
        dict: A dictionary containing the combined data.
    """
    combined_data = {}
    for inner_dict in filtered_data.values():
        for key, value in inner_dict.items():
            if key.startswith('RES_'):
                combined_data[key] = value
    return combined_data


def load_and_filter_data(savefiledir: str, res_analysis: str ='WAT', arR_key: str ="ArR_A") -> dict:
    """
    Load and filter data from files in the specified directory.

    Args:
        savefiledir (str): The directory path where the tmp files of all coordinates are located.
        res_analysis (str, optional): The prefix of the files to filter. Defaults to 'WAT'.
        arR_key (str, optional): The key to use for filtering the data. Defaults to 'ArR_A'.

    Returns:
        dict: A dictionary containing the filtered data, where the keys are the filenames and the values are the filtered data.

    """
    pattern = re.compile(fr'^{res_analysis}.*Block_\d+\.pkl$')
    files_to_filter = [filename for filename in os.listdir(savefiledir) if pattern.match(filename)]

    if not files_to_filter:
        return None

    data_filter = DataFilter(arR_key=arR_key)
    filtered_data = {}

    for filename in files_to_filter:
        file_data = load_file_data(f"{savefiledir}/{filename}")
        filtered_data[filename] = data_filter.process_data_with_filters(file_data)
       

    return combine_data(filtered_data)




def analize_data(trajectory_dir: str, topology: str, analysis: str, slice: int):
    """
    Analyzes trajectory data using the specified analysis function.

    Parameters:
    - trajectory_dir (str): The directory containing the trajectory files.
    - topology (str): The path to the topology file.
    - analysis (function): The analysis function to apply to each frame of the trajectory.
    - slice (int): The offset of frames to analyze.

    Returns:
    - dict: A dictionary containing the concatenated analysis results.

    """
    trajectory_files = natsorted(os.listdir(trajectory_dir))
    trajectory_files = [file for file in trajectory_files if file.endswith("ns.nc")]
    concat_data = {}

    for trajectory_file in trajectory_files:
        traj = pt.iterload(os.path.join(trajectory_dir, trajectory_file), topology, frame_slice=[(0, -1, slice)])
        traj = traj.autoimage()
        print(f"Analyzing {trajectory_file}")

        partial_data = pt.pmap(analysis, traj, n_cores=8)

        for key in partial_data.keys():
            if key in concat_data:
                concat_data[key] = np.vstack((concat_data[key], partial_data[key]))
            else:
                concat_data[key] = partial_data[key]

    return concat_data


def analyze_block(block_key, formatted_residues, trajectory_dir, topology, savefiledir, analysis_resnames, slice):
    """
    Analyzes a block of data from a trajectory and save all extracted coords to a temporary pickle file.

    Parameters:
    - block_key (str): The key of the block to analyze.
    - formatted_residues (list): A list of formatted residues.
    - trajectory_dir (str): The directory containing the trajectory files.
    - topology (str): The path to the topology file.
    - savefiledir (str): The directory to save the analyzed data.
    - analysis_resnames (str): The names of the residues to analyze.
    - slice (slice): The slice of the trajectory to analyze.

    Returns:
    None
    """

    traj_data = analize_data(trajectory_dir, topology, formatted_residues, slice)
    save_data_to_pickle(traj_data, f"{savefiledir}/{analysis_resnames}_coord_Z_{block_key}.pkl")

    
if __name__ == "__main__":
    options = OptionsParser()
    
    topology = options.topofile
    analysis_resnames = options.type
    
    formatted_data = extract_from_topology(topology, f":{analysis_resnames}", int(options.size), pytraj_type="vector center")
    trajectory_dir = options.trajdir
    print("##### ar/R set for MtPIP2;3, dont forget to change if other MIP!#####")

    if not os.path.exists(options.tempdir):
        os.makedirs(options.tempdir)
    savefiledir = options.tempdir

 
    #start analysis from selected block (for restart purposes if needed)
    starting_block = options.start
    start_analysis = False
    clock = TimeControl()
    
    for block_key, formatted_residues in formatted_data.items():
        if block_key == starting_block:
            start_analysis = True
        if not start_analysis:
            continue
        clock.start() 
        print(f"Analyzing {block_key} of {len(formatted_data)}")
        analyze_block(block_key, formatted_residues, trajectory_dir, topology, savefiledir, analysis_resnames, options.slice)
        print(f"Execution time for {block_key}: { round(clock.stop(), 2) } minutes")
        

    for ArR_key in ["ArR_A", "ArR_B", "ArR_C","ArR_D"]:
        print(f"Filtering {ArR_key}")
        clock.start()
        filtered_data = load_and_filter_data(savefiledir, analysis_resnames, ArR_key)
        save_data_to_pickle(filtered_data, f"{analysis_resnames}_coord_Z_{ArR_key}_merged.pkl")
        print(f"Execution time for {ArR_key}: { round(clock.stop(), 2) } minutes")
