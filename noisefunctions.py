import random
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn import preprocessing
from tqdm import tqdm

"""
Functions for generating noise arrays (Fully Modular)
------------------------------------------------------------
Note:
    When a new function is written, do not forget to add to noise_dict dictionary.
"""

def gauss_noise(df, key_col, std, seed):
    """
    Helper function that generates a gaussian curve of noises array
    
    Parameters:
        df: pandas dataframe
            Dataframe to read from
        key_col: string
            The key column of the property values to add noise to
        std: float
            Standard deviation for the gaussian curve to add noise from
        seed: int
            Seed for randomness
        
    Returns:
        An array of gaussian noises
    """
    #Allows for reproducibility
    np.random.seed(seed)

    noise = np.random.normal(0, std, [len(df[key_col]),1], )
    return np.concatenate(noise)    

def const_bias(df, key_col, const, seed):
    """
    Helper function that generates constant noises (biases) array
    
    Parameters:
        df: pandas dataframe
            Dataframe to read from
        key_col: string
            The key column of the property values to add noise to
        const: float
            Constant value for the noise (bias)
        seed: int
            Not used
        
    Returns:
        An array of constant noises
    """
    noise = [float(const) for index in range(len(df[key_col]))]
    return noise

def uniform_rand_noise(df, key_col, rand_range, seed):
    """
    Helper function that generates a purely random noise array
    
    Parameters:
        df: pandas dataframe
            Dataframe to read from
        key_col: string
            The key column of the property values to add noise to
        rand_range: list
            Range of allowed random numbers
        seed: int
            Seed for randomness   
        
    Returns:
        An array of random noises within a range
    """
    #Allows for reproducibility
    random.seed(seed)

    noise = [random.uniform(rand_range[0],rand_range[1]) for index in range(len(df[key_col]))]
    return noise

def descriptor_bias(df, descriptors):
    """
    Given a series of descriptors and corresponding coefficients,
    calculates the weighted sum and returns it as a list of biases

    Parameters:
        df: pandas dataframe
            Dataframe to read from
        *descriptors: tuple
            Variable amount of tuples of descriptor and weight in bias
    
    Returns:
        Numpy array of biases calculated from the weighted sum of the descriptor
        values
    """
    # Get mols objects from all the smiles indexes
    mols = list(map(Chem.MolFromSmiles, df.index))
    # Initialize the resulting array
    result = np.zeros((1,len(df.index)))
    # Iterate over the descriptor/coefficient pairs, normalize and add the 
    # weighted result to the result array
    for descriptor, coefficient in tqdm(descriptors):
        pre_result = tuple(map(descriptor, mols.copy()))
        result += np.array(preprocessing.normalize([pre_result]) * coefficient) 

    return result
    
global noise_dict

noise_dict = {
        'constant':const_bias,
        'gauss':gauss_noise,
        'random':uniform_rand_noise
    }
