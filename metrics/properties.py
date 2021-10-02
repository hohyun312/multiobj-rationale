from metrics import sascorer
from rdkit.Chem import Descriptors
    

def get_scoring_function(prop_name):
    prop_name = prop_name.lower()
    if prop_name == 'qed':
        return Descriptors.qed
    elif prop_name == 'sa':
        return sascorer.calculateScore
    elif prop_name == 'logp':
        return Descriptors.MolLogP
    elif prop_name == 'mw':
        return Descriptors.MolWt
    else:
        raise ValueError('unsupported scoring function')