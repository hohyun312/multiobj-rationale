from collections import deque
import random

from rdkit.Chem.rdmolfiles import SmilesMolSupplier
from rdkit.Chem.rdchem import BondType
from rdkit import Chem
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
        has_header = bool("smiles" in first_line.lower())

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

    def __init__(self, atom_types, use_aromatic_bonds=False):
        self.use_aromatic_bonds = use_aromatic_bonds
        
        # define edge types
        self.bondtype_to_int = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2}
    
        if use_aromatic_bonds:
            self.bondtype_to_int[BondType.AROMATIC] = 3
        
        self.int_to_bondtype = {v: k for k, v in self.bondtype_to_int.items()}    
        
        # define node types
        self.atomtype_to_int = {a:i for i, a in enumerate(atom_types)}
        self.int_to_atomtype = {v: k for k, v in self.atomtype_to_int.items()}
        
        # number of types
        self.n_atom_types = len(self.atomtype_to_int)
        self.n_edge_types = len(self.bondtype_to_int)
        
        
        
# def mol_to_graph(mol):

#     if not vocab.use_aromatic_bonds:
#         Chem.Kekulize(mol, clearAromaticFlags=True)

#     node_types = []
#     edge_u_idx = []
#     edge_v_idx = []
#     edge_types = []

#     queue = deque([0])
#     node_mapping = {0:0}
#     while queue:
#         cur_idx = queue.popleft()

#         atom = mol.GetAtomWithIdx(cur_idx)
#         node_types.append((atom.GetSymbol(), atom.GetFormalCharge()))

#         for nei in atom.GetNeighbors():
#             nei_idx = nei.GetIdx()
#             if nei_idx not in node_mapping:
#                 node_mapping[nei_idx] = len(node_mapping)

#                 bond = mol.GetBondBetweenAtoms(cur_idx, nei_idx)
#                 edge_u_idx.append(node_mapping[cur_idx])
#                 edge_v_idx.append(node_mapping[nei_idx])
#                 edge_types.append(bond.GetBondType())

#                 for neinei in nei.GetNeighbors():
#                     neinei_idx = neinei.GetIdx()
#                     if neinei_idx in queue:
#                         bond = mol.GetBondBetweenAtoms(nei_idx, neinei_idx)
#                         edge_u_idx.append(node_mapping[nei_idx])
#                         edge_v_idx.append(node_mapping[neinei_idx])
#                         edge_types.append(bond.GetBondType())

#                 queue.append(nei_idx)

#     node_types = torch.LongTensor([vocab.atomtype_to_int[n] for n in node_types])
#     edge_index = torch.LongTensor([edge_u_idx, edge_v_idx])
#     edge_types = torch.LongTensor([vocab.bondtype_to_int[e] for e in edge_types])

#     return MolGraph(node_types, edge_index, edge_types)

def mol_to_graph(mol):

    if not vocab.use_aromatic_bonds:
        Chem.Kekulize(mol, clearAromaticFlags=True)

    node_types = []
    edge_u_idx = []
    edge_v_idx = []
    edge_types = []

    queue = deque([0])
    node_mapping = {}
    while queue:
        idx = queue.popleft()
        if idx not in node_mapping:
            node_mapping[idx] = len(node_mapping)

            atom = mol.GetAtomWithIdx(idx)
            node_types.append((atom.GetSymbol(), atom.GetFormalCharge()))

            neighbors = [nei.GetIdx() for nei in atom.GetNeighbors()]
            neighbors = sorted(neighbors, key=lambda x: node_mapping.get(x, 9999))
            
            for nei in neighbors:
                if nei in node_mapping:
                    bond = mol.GetBondBetweenAtoms(idx, nei)
                    edge_u_idx.append(node_mapping[idx])
                    edge_v_idx.append(node_mapping[nei])
                    edge_types.append(bond.GetBondType())
                    
            queue.extend(neighbors)

    node_types = torch.LongTensor([vocab.atomtype_to_int[n] for n in node_types])
    edge_index = torch.LongTensor([edge_v_idx, edge_u_idx])
    edge_types = torch.LongTensor([vocab.bondtype_to_int[e] for e in edge_types])

    return MolGraph(node_types, edge_index, edge_types)
        
    
class MolGraph:
    def __init__(self, node_types, edge_index, edge_types):
        
        self.node_types = node_types
        self.edge_index = edge_index
        self.edge_types = edge_types
        
        self.gs  = [self]
        self.ptr = [0, self.num_nodes()]

        self.device = "cpu"

    def set_device(self, device):
        self.device = device
        self.node_types = self.node_types.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_types = self.edge_types.to(device)
        
    def num_nodes(self):
        return len(self.node_types)
    
    def num_edges(self):
        return len(self.edge_types)
    
    def num_mols(self):
        return len(self.gs)
            
    def to_tensors(self):
        return self.node_types, self.edge_index, self.edge_types
        
    def to_adjacency(self):
        
        n_nodes = self.num_nodes()
        n_edges = self.num_edges()
        
        # undirected
        indices = torch.vstack(
            [
                torch.cat([self.edge_types, self.edge_types]),
                torch.cat([self.edge_index[1], self.edge_index[0]]),
                torch.cat([self.edge_index[0], self.edge_index[1]])
            ])
        adj = torch.sparse_coo_tensor(indices=indices, 
                                values=torch.ones(2*n_edges), 
                                size=(vocab.n_edge_types, n_nodes, n_nodes))
        return adj
    
    def to_mol(self):    
        empty_mol = Chem.rdchem.Mol()
        new_mol = Chem.RWMol(empty_mol)
        for node_idx in self.node_types:
            atom_symbol, charge = vocab.int_to_atomtype[node_idx.item()]
            atom = Chem.rdchem.Atom(atom_symbol)
            atom.SetFormalCharge(charge)
            new_mol.AddAtom(atom)

        for eid in range(self.num_edges()):
            e_i = self.edge_index[0, eid].item()
            e_j = self.edge_index[1, eid].item()
            edge_idx = self.edge_types[eid].item()
            bondtype = vocab.int_to_bondtype[edge_idx]
            new_mol.AddBond(e_i, e_j, bondtype)

        return new_mol.GetMol()
    
    def random_subgraph(self, min_atoms=10):
        sub_gs = []
        for g in self.gs:
            min_atoms = min(min_atoms, g.num_nodes())
            n_atoms = random.randint(min_atoms, g.num_nodes())

            cond = (g.edge_index < n_atoms)
            mask = cond[0] & cond[1]

            edge_index = g.edge_index[:,mask]
            node_types = g.node_types[:n_atoms]
            edge_types = g.edge_types[:len(edge_index[0])]

            sub_gs.append( MolGraph(node_types, edge_index, edge_types)  )

        return sum(sub_gs)
    
    def adj(self, i):
        '''
        {adjacent node: edge_type}
        '''
        mask0 = self.edge_index[0] == i
        mask1 = self.edge_index[1] == i

        ei = self.edge_index[1, mask0].numpy().tolist() + self.edge_index[0, mask1].numpy().tolist()
        et = self.edge_types[mask0].numpy().tolist() + self.edge_types[mask1].numpy().tolist()
        return dict(zip(ei, et))

    
    def __radd__(self, g):
        if isinstance(g, int):
            return self
        
        offset = self.num_nodes()
        
        node_types = torch.cat([self.node_types, g.node_types], dim = 0)
        edge_types = torch.cat([self.edge_types, g.edge_types], dim = 0)
        edge_index = torch.cat([self.edge_index, offset+g.edge_index], dim = 1)
        mol_graph = MolGraph(node_types, edge_index, edge_types)
        
        mol_graph.gs = self.gs + [g]
        mol_graph.ptr = self.ptr + [offset + g.num_nodes()]
        
        return mol_graph
    
    def __getitem__(self, index):
        return self.gs[index]
    
    def __iter__(self):
        return iter(self.gs)
    
    def __add__(self, g):
        return self.__radd__(g)
    
    def __repr__(self):
        return "<Mol:%s, N:%s, E:%s>" % (self.num_mols(), self.num_nodes(), self.num_edges())
    
    
class LabelGenerator():
    def __init__(self, full_graph, sub_graph):
        self.full_graph = full_graph
        self.sub_graph  = sub_graph
        
        self.front_nodes, self.next_nodes = self._get_queued_node_index()
        self.next_node_neighbors = [full_g.adj(next_n) 
                 for full_g, next_n in zip(full_graph, self.next_nodes.numpy())]
    
    def front_index_and_node_labels(self):
        # get front index
        node_front_index = self.front_nodes + torch.LongTensor(self.sub_graph.ptr[:-1])
        
        # get node label
        atom_labels = []
        n_nodes   = self.next_nodes.numpy()
        f_nodes   = self.front_nodes.numpy()
        nei_nodes = self.next_node_neighbors # List[dict]
        
        for i in range(len(f_nodes)):
            if f_nodes[i] in nei_nodes[i]:
                # if next node is adjacent to the front node, retrieve next node's type
                y = self.full_graph[i].node_types[n_nodes[i]]
            else:
                y = vocab.n_atom_types
            atom_labels.append(y)
        
        return node_front_index, torch.LongTensor(atom_labels)
    
    def queue_index_and_edge_labels(self, mask):
        # get queue_index
        front_nodes = self.front_nodes[mask]
        next_nodes  = self.next_nodes[mask]
        sequences = [list(range(i, j)) 
                     for i, j in zip(front_nodes, next_nodes)]
        
        # get edge label
        next_node_neighbors = [self.next_node_neighbors[i] for i, m in enumerate(mask) if m]
        pad_id = vocab.n_edge_types
        bond_labels = []
        for seq, nnn in zip(sequences, next_node_neighbors):
            bond_labels.append([nnn.get(node, pad_id) for node in seq])
        return (self._pack_list_sequence(sequences, -1),
                self._pack_list_sequence(bond_labels, pad_id)
               )
    
    def _get_queued_node_index(self):
        front_nodes = []
        next_nodes = []
        Q = []
        for g in self.sub_graph:
            q_id = torch.argmax(g.edge_index[1], 0)
            q_min = g.edge_index[0, q_id].item()
            q_max = g.num_nodes()
            front_nodes.append(q_min)
            next_nodes.append(q_max)
        
        front_nodes = torch.LongTensor(front_nodes)
        next_nodes  = torch.LongTensor(next_nodes)
        return front_nodes, next_nodes
    
    def _pack_sequence(self, sequences, pad_id=0):
        maxlen = max(map(lambda x: len(x), sequences))
        padded = [torch.cat([seq, torch.LongTensor([pad_id]*(maxlen-len(seq)))]) 
                  for seq in sequences]
        return torch.vstack(padded)
    
    def _pack_list_sequence(self, sequences, pad_id=0):
        maxlen = max(map(lambda x: len(x), sequences))
        padded = [seq+[pad_id]*(maxlen-len(seq)) for seq in sequences]
        return torch.LongTensor(padded)
    

atom_types = [('B', 0), ('B', -1), ('Br', 0), ('Br', -1), ('Br', 2), ('C', 0), ('C', 1), ('C', -1), ('Cl', 0), ('Cl', 1), ('Cl', -1), ('Cl', 2), ('Cl', 3), ('F', 0), ('F', 1), ('F', -1), ('I', -1), ('I', 0), ('I', 1), ('I', 2), ('I', 3), ('N', 0), ('N', 1), ('N', -1), ('O', 0), ('O', 1), ('O', -1), ('P', 0), ('P', 1), ('P', -1), ('S', 0), ('S', 1), ('S', -1), ('Se', 0), ('Se', 1), ('Se', -1), ('Si', 0), ('Si', -1)]

vocab = Vocab(atom_types, use_aromatic_bonds=False)