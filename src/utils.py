# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Authors: Francisco Melícias e Tiago F. R. Ribeiro
# Creation date (file creation): 24/10/2023
# Description: This file contains utility functions used in the project.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pandas as pd

def save_results_to_csv(results, csv_file_path):
    """
    Saves the results to a CSV file.    
        
    Parameters:
        results : dict 
            Dictionary containing the results.
        csv_file_path : str
            Path to the CSV file.
    """
    
    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    # Check if the file exists
    file_exists = False
    try:
        pd.read_csv(csv_file_path)
        file_exists = True
    except FileNotFoundError:
        pass

    # Append results to the CSV file
    df.to_csv(csv_file_path, mode='a', header=not file_exists, index=False)