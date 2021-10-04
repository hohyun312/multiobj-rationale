import os
import pickle
import numpy as np
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
from metrics import sascorer


class GSK3BModel():
    """Scores based on an ECFP classifier for activity."""

    clf_path = os.path.join(os.path.dirname(__file__), 'gsk3.pkl')

    def __init__(self):
        with open(self.clf_path, "rb") as f:
            self.clf = pickle.load(f)

    def __call__(self, mol):
        fp = GSK3BModel.fingerprints_from_mol(mol)
        return self.clf.predict_proba(fp)[0, 1]

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)

    
class JNK3Model():
    """Scores based on an ECFP classifier for activity."""
    
    clf_path = os.path.join(os.path.dirname(__file__), 'jnk3.pkl')

    def __init__(self):
        with open(self.clf_path, "rb") as f:
            self.clf = pickle.load(f)

    def __call__(self, mol):
        fp = JNK3Model.fingerprints_from_mol(mol)
        return self.clf.predict_proba(fp)[0, 1]

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)
    

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
    elif prop_name == 'gsk3b':
        return GSK3BModel()
    elif prop_name == 'jnk3':
        return JNK3Model()
    else:
        raise ValueError('unsupported scoring function')