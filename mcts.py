from rdkit import Chem
import math


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
    # (atom.GetBonds()[0].GetBondType() is not Chem.rdchem.BondType.AROMATIC) # only include nonaromatic bond

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
        self.N = 1     # visit count
        self.P = P     # predicted property score
        
        if smiles:
            self.smi = smiles
        else:
            self.smi = Chem.MolToSmiles(self.mol)
                        
    def Q(self):
        return self.W / self.N if self.N > 0 else 0

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
    Chem.Kekulize(mol, clearAromaticFlags=True)
    root = MCTSNode(mol, smiles)
    
    state_map = {}
    for _ in range(n_rollout):
        mcts_rollout(root, scoring_function, state_map)

    nodes = [node for _, node in state_map.items() if node.mol.GetNumAtoms() <= max_atoms and node.P >= prop_delta]
    rationales = [n.smi for n in nodes]
    scores = [n.P for n in nodes]
    
    return rationales, scores


if __name__ == '__main__':
    
    import argparse
    from metrics import get_scoring_function
    
    
    parser = argparse.ArgumentParser()
#     parser.add_argument('--data', required=True)
    parser.add_argument('--prop', type=str, default='qed')
    parser.add_argument('--n_rollout', type=int, default=5)
    parser.add_argument('--c_puct', type=float, default=10)
    parser.add_argument('--max_atoms', type=int, default=20)
    parser.add_argument('--min_atoms', type=int, default=15)
    parser.add_argument('--prop_delta', type=float, default=0.5)
    args = parser.parse_args()
    
    C_PUCT = args.c_puct
    MIN_ATOMS = args.min_atoms
    
#     with (args.data, 'r') as f:
#         data = r.read()
    data = """CC(=O)NCCC1=CNc2c1cc(OC)cc2CC(=O)NCCc1c[nH]c2ccc(OC)cc12
    O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5
    OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H]2[C@@H]1c3c(O)c(OC)c(O)cc3C(=O)O2
    OCCc1c(C)[n+](cs1)Cc2cnc(C)nc2N
    COC(=O)[C@H](CCCCN)NC(=O)Nc1cc(OC)cc(C(C)(C)C)c1O
    """.split() # => sample data for testing


    print("smiles,rationale,score")
    scoring_func = get_scoring_function(args.prop)

    for smi in data:
        z = mcts(smi, scoring_func, n_rollout=args.n_rollout, max_atoms=args.max_atoms, prop_delta=args.prop_delta)
        for rationale, score in zip(*z):
            print("%s,%s,%s" %(smi, rationale, score))