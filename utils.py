from rdkit import Chem


def sanitize_mol(mol):
    """
    True if no sanity issues, False otherwise
    """
    try:
        Chem.SanitizeMol(mol)
        return True
    except ValueError:
        return False