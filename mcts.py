import math
from rdkit import Chem
from utils import sanitize_mol


def remove_atoms(mol, atoms):
    mw = Chem.RWMol(mol)
    for atom in sorted(atoms, reverse=True):
        mw.RemoveAtom(atom)
    return mw.GetMol()
    

def get_leaves(mol):
    """
    Finds peripheral atom or ring within the molecule.
    """    
    leaf_atoms = [[atom.GetIdx()] for atom in mol.GetAtoms() if (atom.GetDegree() == 1)]

    clusters = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            clusters.append( set([a1,a2]) )

    rings = [set(x) for x in Chem.GetSymmSSSR(mol)]
    clusters.extend(rings)
    
    leaf_rings = []
    for r in rings:
        inters = [c for c in clusters if r != c and len(r & c) > 0]
        if len(inters) > 1: continue
        nodes = [i for i in r if mol.GetAtomWithIdx(i).GetDegree() == 2]
        leaf_rings.append(nodes)

    return leaf_atoms + leaf_rings


class MCTSNode:
    
    def __init__(self, mol, smiles=None, P=0):
        self.mol = mol
        self.children = []
        self.W = 0     # total action value
        self.N = 0     # visit count
        self.P = P     # predicted property score
        
        if smiles:
            self.smi = smiles
        else:
            self.smi = Chem.MolToSmiles(self.mol)
                        
    def Q(self):
        return self.W / self.N if self.N > 0 else float("inf")

    def U(self, n):
        return C_PUCT * self.P * math.sqrt(n) / (1 + self.N)
    
    def __repr__(self):
        return '<MCTSNode: %s>' % self.smi
    
    
def mcts_rollout(node, scoring_function, state_map):

    if node.mol.GetNumAtoms() <= MIN_ATOMS:
        return node.P

    # Expand if this node has never been visited
    if len(node.children) == 0:
        leaves = get_leaves(node.mol)
        for leaf in leaves:
            new_mol = remove_atoms(node.mol, leaf)
            # check sanity
            if not sanitize_mol(new_mol): continue
                
            new_smi = Chem.MolToSmiles(new_mol)
            if new_smi in state_map:
                new_node = state_map[new_smi]  # merge identical states
            else:
                new_node = MCTSNode(new_mol, new_smi, P=scoring_function(new_mol))
            
            node.children.append(new_node)
        
        state_map[node.smi] = node
        if len(node.children) == 0:
            return node.P  # cannot find leaves
        
    sum_count = sum(c.N for c in node.children)
    selected_node = max(node.children, key=lambda x: x.Q() + x.U(sum_count))
    v = mcts_rollout(selected_node, scoring_function, state_map)
    selected_node.W += v
    selected_node.N += 1

    return v


def mcts(smiles, scoring_function, n_rollout, max_atoms, prop_delta):
            
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    root = MCTSNode(mol, smiles)
    
    state_map = {}
    for _ in range(n_rollout):
        mcts_rollout(root, scoring_function, state_map)

    data = [(node.smi, node.P) for _, node in state_map.items() if node.mol.GetNumAtoms() <= max_atoms and node.P >= prop_delta]
    
    return data


def read_active_data(filepath):
    smiles_list = []
    with open(filepath, 'r') as f:
        next(f)
        while True:
            line = f.readline().rstrip('\n')
            if line:
                smiles, active = line.split(',')
                if active == '1':
                    smiles_list.append(smiles)
            else:
                break
    return smiles_list


if __name__ == '__main__':
    
    import argparse
    import rdkit
    from metrics import get_scoring_function
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL) # mute warnings
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--prop', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--n_rollout', type=int, default=20)
    parser.add_argument('--c_puct', type=float, default=10)
    parser.add_argument('--max_atoms', type=int, default=20)
    parser.add_argument('--min_atoms', type=int, default=15)
    parser.add_argument('--prop_delta', type=float, default=0.5)
    args = parser.parse_args()
    
    C_PUCT = args.c_puct
    MIN_ATOMS = args.min_atoms
    
    if args.prop == 'jnk3':
        print("processing jnk3 data...")
        scoring_func = get_scoring_function('jnk3')
        actives = read_active_data("./data/excape-db/jnk3.csv") 
    elif args.prop == 'gsk3b':
        print("processing gsk3b data...")
        scoring_func = get_scoring_function('gsk3b')
        actives = read_active_data("./data/excape-db/gsk3b.csv") 
    else:
        raise ValueError("--prop must be set either to jnk3 or gsk3b")
        
        
    

    with open(args.out, 'w') as f: 
        f.write("original_smiles,rationale_smiles,score\n")

        for i, smi in enumerate(actives,1):
            print('(%s/%s)'%(i, len(actives)), smi)
            z = mcts(smi, scoring_func, n_rollout=args.n_rollout, max_atoms=args.max_atoms, prop_delta=args.prop_delta)
            if z is not None:
                for rationale, score in z:
                    f.write("%s,%s,%s\n" %(smi, rationale, score))
