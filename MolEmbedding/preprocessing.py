import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdchem import Atom
import matplotlib.pyplot as plt
import pickle

def get_max_seq_len(input_smiles_file, input_char_file):
    # load chemical symbols
    with open(input_char_file, 'rb') as fp:
        char_list = pickle.load(fp)
    fp.close()

    with open(input_smiles_file) as fp:
        smiles = fp.read().splitlines()

    smiles_len_lst=[]

    for s in tqdm(smiles):
        tokens_lst = smiles_to_tokens(s, char_list)
        smiles_len_lst.append(len(tokens_lst))
    fp.close()

    smiles_len_array = np.asarray(smiles_len_lst)
    return smiles_len_array


def smiles_to_tokens(s, char_list):
    s = s.strip()

    j = 0
    tokens_lst = []
    while j < len(s):
        # handle atoms with two characters
        if j < len(s) - 1 and s[j:j + 2] in char_list:
            token = s[j:j + 2]
            j = j + 2

        # handle atoms with one character including hydrogen
        elif s[j] in char_list:
            token = s[j]
            j = j + 1

        # handle unknown characters
        else:
            token = '<unk>'
            j = j + 1

        tokens_lst.append(token)

    return tokens_lst

def smiles_to_mol(smiles):
    """
    convert smiles to a rdkit molecule object
    :param smiles: input smiles
    :return: rdkit molecule object, or None when smiles is invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except:
        return None

def mol_to_atoms(mol):
    """
    get the sequence of atoms in smiles
    :param mol: molecule object
    :return: unique heavy atomic symbols
    """
    atom_lst = []
    for at in mol.GetAtoms():
        atom_lst.append(Atom.GetSymbol(at))
    return atom_lst

def smiles_file_clean(input_smiles_file, chem_symbol_file, spec_char_file):
    """
    1, clean input smiles file, remove invalid smiles and convert smiles to canonical format
    2, extract unique tokens (chemical symbols and special characters from the cleaned smiles file)
    3, save cleaned smiles file and token/character file

    :param input_smiles_file: raw input smiles file
    :param chem_symbol_file: chemical symbols file
    :param spec_char_file: special character file
    :return:
    """

    # load chemical symbols
    with open(chem_symbol_file, 'rb') as fp:
        chem_elements = pickle.load(fp)
    fp.close()

    # load special characters
    with open(spec_char_file, 'rb') as fp:
        spec_char = pickle.load(fp)
    fp.close()

    # process input smiles file
    with open(input_smiles_file) as fp:
        smiles = fp.read().splitlines()

    atoms_lst = []
    spec_lst = []
    miss_lst = []
    smiles_lst_valid = []

    for s in tqdm(smiles):
        s = s.strip()
        mol = smiles_to_mol(s)
        try:
            atoms = set(mol_to_atoms(mol))
            # convert smiles to canonical format
            smiles_can = Chem.MolToSmiles(mol)

            # add cleaned smiles
            smiles_lst_valid.append(smiles_can)

            j = 0
            while j < len(s):
                # handle atoms with two characters
                if j < len(s) - 1 and s[j:j + 2] in atoms:
                    atoms_lst.append(s[j:j + 2])
                    j = j + 2

                # handle atoms with one character including hydrogen
                elif s[j] in chem_elements:
                    atoms_lst.append(s[j])
                    j = j + 1

                # handle special characters
                elif s[j] in spec_char:
                    spec_lst.append(s[j])
                    j = j + 1

                # capture other unexpected characters
                else:
                    miss_lst.append(s[j])
                    j = j + 1
        except:
            pass
    fp.close()

    # save cleaned smiles file and distinct characters
    #atoms_set = set(atoms_lst)
    #spec_set = set(spec_lst)
    #miss_set = set(miss_lst)
    char_set = set(atoms_lst + spec_lst + miss_lst)
    smiles_set_valid = set(smiles_lst_valid)

    with open('./data/smiles_set_clean.smi', 'w') as fp:
        for s in smiles_set_valid:
            fp.write('%s\n' % s)
    fp.close()

    with open('./data/char_set_clean.pkl', 'wb') as fp:
        pickle.dump(list(char_set), fp)
    fp.close()


"""
# step 1: create master character lists, including all chemical elements and special characters
# all chemical elements in the FDA approved drugs
chem_elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P'
                 , 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu'
                 , 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc'
                 , 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La'
                 , 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'
                 , 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At'
                 , 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es'
                 , 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

spec_char = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
             , ' ', '-', '#', '(', ')', '[', ']', '+', '=', '#', '/', '@', '\\', '.', '%'
             , 'c', 'n', 'o', 'p', 's']

char_set_master = chem_elements + spec_char

# save to files
with open('./data/chem_elements.pkl', 'wb') as fp:
    pickle.dump(chem_elements, fp)
fp.close()

with open('./data/spec_char.pkl', 'wb') as fp:
    pickle.dump(spec_char, fp)
fp.close()


# step 2: find the unique set of characters from the input SMILES file
chem_symbol_file = './data/chem_elements.pkl'
spec_char_file = './data/spec_char.pkl'
input_smiles_file = './data/250k_rndm_zinc_drugs_clean.smi'
#input_smiles_file = './data/all_drugs_only.smi'
#input_smiles_file = '/Users/keyu/Documents/MolEmbedding/data/pubchem_1_5M.txt'
#input_smiles_file = './data/all_drugs_zinc.smi'
#input_smiles_file = './data/pubchem_clean/smiles_set_clean.smi'

# clean input smiles files, removing invalid smiels; extract unique token file
smiles_file_clean(input_smiles_file, chem_symbol_file, spec_char_file)
"""

# distribution of smiles lens
input_smiles_file = './data/zinc_fda_drugs_clean/smiles_set_clean.smi'
input_char_file = './data/zinc_fda_drugs_clean/char_set_clean.pkl'
smiles_len_array = get_max_seq_len(input_smiles_file, input_char_file)
stop = 0