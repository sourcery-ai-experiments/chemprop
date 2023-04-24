import random
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors
from scaffold_splits import scaffold_split

"""
Functions for calculating the property value (Fully Modular)
------------------------------------------------------------
Note:
    When a new function is written, do not forget to add to split_by_prop_dict dictionary.
"""


def molwt_split(df, cutoff_val: float = 150, seed: int = 0):
    """
     Function to split the dataframe into two sets where one is above the cutoff molecular weight and the other is below

    Parameters:
        df: pandas dataframe
            Dataframe to fetch the smiles molecule
        cutoff_val: float
            Threshold value for splitting

    Returns:
        A tuple of lists containing the smiles representation and the corresponding molecular weight
    """
    # Allow for reproducibility
    random.seed(seed)

    # Initialize array
    test, train_val = [], []

    for entry in tqdm(df.index):

        # Calculate molecular weight
        mol = Chem.MolFromSmiles(entry)
        entry_molwt = Descriptors.ExactMolWt(mol)

        # If the property value of the current entry has a value over the cutoff then add it to the potential train_val set
        if entry_molwt >= cutoff_val:
            train_val.append(entry)
        # Otherwise send it to the potential test set
        else:
            test.append(entry)

    random.shuffle(train_val)
    random.shuffle(test)

    return (train_val, test)


def h298_split(df, cutoff_val: float = 0, seed: int = 0):
    """
    Function to split the dataframe into two sets where one is above the cutoff h298 value and the other is below

    Parameters:
        df: pandas dataframe
            Dataframe to find the h298 value of the given smiles molecule
        cutoff_val: float
            Threshold value for splitting

    Returns:
        A tuple of lists containing the smiles representation and the corresponding h298 value
    """
    # Allow for reproducibility
    random.seed(seed)

    col_name = df.columns[0]

    # Initialize array
    test, train_val = [], []

    for index in tqdm(df.index):

        # If the enthalpy of the current entry has a value over the cutoff then add it to the potential train_val set
        if df[col_name][index] >= cutoff_val:
            train_val.append(index)
        # Otherwise send it to the potential test set
        else:
            test.append(index)

    random.shuffle(train_val)
    random.shuffle(test)

    return (train_val, test)


def atoms_to_exclude(df, restricted_list: list = ["F"], seed: int = 0):
    """
    Function to split the dataframe into two sets where one is above the cutoff h298 value and the other is below

    Parameters:
        df: pandas dataframe
            Dataframe to find the h298 value of the given smiles molecule
        cutoff_val: list
            List containing disallowed atoms

    Returns:
        A tuple of lists containing smiles representations
    """
    # Allow for reproducibility
    random.seed(seed)

    # Initialize array
    test, train_val = [], []

    # Initialize symbol to atomic number dictionary (Expandable)
    atom_dict = {"B": 5, "N": 7, "O": 8, "F": 9}

    restricted = restricted_list[:]
    # Translate list of disallowed atom strings into a list of atomic numbers
    for index in range(len(restricted)):
        restricted[index] = atom_dict[restricted[index]]

    # Flag used for when atomic_num was in the list and the current entry need not be appended
    # to the train_val as it is already in test
    flag = False

    for entry in tqdm(df.index):
        flag = False
        mol = Chem.MolFromSmiles(entry)
        # Construct the molecule and iterate atom after atom
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()

            # If it matches any disallowed atomic numbers then append to test and move on to next iteration
            if atomic_num in restricted:
                test.append(entry)
                flag = True
                break
        # Otherwise flag will be false and it can appended to train_val
        if not flag:
            train_val.append(entry)

    random.shuffle(train_val)
    random.shuffle(test)

    return (train_val, test)


def random_split(df, sizes: tuple = (0.9, 0.1), seed: int = 0):
    """
    Function to split the dataframe into two sets where one is above the cutoff h298 value and the other is below

    Parameters:
        df: pandas dataframe
            Dataframe to split randomly
        sizes: tuple
            Proportions to split into

    Returns:
        A tuple of lists containing smiles representations
    """
    assert sum(sizes) == 1

    train_val = list(df.sample(frac=sizes[0], random_state=seed).index)
    test = list(df.drop(train_val).index)

    random.shuffle(train_val)
    random.shuffle(test)

    return (train_val, test)

global split_by_prop_dict

split_by_prop_dict = {
    "h298": h298_split,
    "molwt": molwt_split,
    "atom": atoms_to_exclude,
    "random": random_split,
    "scaffold": scaffold_split,
}
