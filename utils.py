from rdkit.Chem.rdmolfiles import SmilesMolSupplier
from rdkit.Chem.rdchem import BondType
from rdkit import Chem
from collections import deque
import torch


def sanitize_mol(mol):
    """
    True if no sanity issues, False otherwise
    """
    try:
        Chem.SanitizeMol(mol)
        return True
    except ValueError:
        return False
    
    
def read_molecules(path):
    """
    Reads a SMILES file and returns `rdkit.Mol` objects.
    """
    # check first line of SMILES file to see if contains header
    with open(path) as smi_file:
        first_line = smi_file.readline()
        has_header = bool("SMILES" in first_line)
    smi_file.close()

    # read file
    molecule_set = SmilesMolSupplier(path,
                                     sanitize=True,
                                     nameColumn=-1,
                                     titleLine=has_header)
    return molecule_set
    

def get_smiles(molecule):
    try:
        smiles = Chem.MolToSmiles(mol=molecule, kekuleSmiles=False)
    except:
        # if molecule is invalid, set SMILES to `None`
        smiles = None
    return smiles


class Vocab:
    '''
    Convert molecules to sparse tensor format, and vice versa.
    '''
    def __init__(self, atom_types, n_node_features=None, use_aromatic_bonds=False):
        self.atom_types         = atom_types
        self.use_aromatic_bonds = use_aromatic_bonds
        
        # define edge feature (rdkit `GetBondType()` result -> `int`) constants
        self.bondtype_to_int = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2}
    
        if use_aromatic_bonds:
            self.bondtype_to_int[BondType.AROMATIC] = 3
        
        self.int_to_bondtype = {v: k for k, v in self.bondtype_to_int.items()}    
        self.n_edge_types = len(self.bondtype_to_int)
        
        # define node feature
        self.atomtype_to_int = {a:i for i, a in enumerate(self.atom_types)}
        self.int_to_atomtype = {v: k for k, v in self.atomtype_to_int.items()}
        
        if n_node_features is None:
            self.n_node_features = len(atom_types)
        else:
            self.n_node_features = n_node_features
        
        
    def mol_to_tensor(self, mol):
        
        if mol is None:
            return None
            
        if not self.use_aromatic_bonds:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        
        node_types = []
        edge_u_idx = []
        edge_v_idx = []
        edge_types = []

        queue = deque([0])
        node_visited = {}
        while queue:
            idx = queue.popleft()
            if idx not in node_visited:
                node_visited[idx] = len(node_visited)

                atom = mol.GetAtomWithIdx(idx)
                node_types.append((atom.GetSymbol(), atom.GetFormalCharge()))

                neighbors = [nei.GetIdx() for nei in atom.GetNeighbors()]

                for nei in neighbors:
                    if nei in node_visited:
                        edge_u_idx.append(node_visited[idx])
                        edge_v_idx.append(node_visited[nei])
                        bond = mol.GetBondBetweenAtoms(idx, nei)
                        edge_types.append(bond.GetBondType())


                queue.extend(neighbors)

        node_types = torch.LongTensor([self.atomtype_to_int[n] for n in node_types])
        edge_index = torch.LongTensor([edge_u_idx, edge_v_idx])
        edge_types = torch.LongTensor([self.bondtype_to_int[e] for e in edge_types])

        return node_types, edge_index, edge_types
    
    
    def tensor_to_mol(self, node_types, edge_index, edge_types):
        empty_mol = Chem.rdchem.Mol()
        new_mol = Chem.RWMol(empty_mol)
        for node_idx in node_types:
            atom_symbol, charge = self.int_to_atomtype[node_idx.item()]
            atom = Chem.rdchem.Atom(atom_symbol)
            atom.SetFormalCharge(charge)
            new_mol.AddAtom(atom)

        for eid in range(len(edge_types)):
            e_i = edge_index[0, eid].item()
            e_j = edge_index[1, eid].item()
            edge_idx = edge_types[eid].item()
            bondtype = self.int_to_bondtype[edge_idx]
            new_mol.AddBond(e_i, e_j, bondtype)

        return new_mol.GetMol()
    
    
atom_types = [('B', 0), ('B', -1), ('Br', 0), ('Br', -1), ('Br', 2), ('C', 0), ('C', 1), ('C', -1), ('Cl', 0), ('Cl', 1), ('Cl', -1), ('Cl', 2), ('Cl', 3), ('F', 0), ('F', 1), ('F', -1), ('I', -1), ('I', 0), ('I', 1), ('I', 2), ('I', 3), ('N', 0), ('N', 1), ('N', -1), ('O', 0), ('O', 1), ('O', -1), ('P', 0), ('P', 1), ('P', -1), ('S', 0), ('S', 1), ('S', -1), ('Se', 0), ('Se', 1), ('Se', -1), ('Si', 0), ('Si', -1)]

vocab = Vocab(atom_types, n_node_features=None, use_aromatic_bonds=False)