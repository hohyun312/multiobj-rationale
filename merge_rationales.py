from rdkit import Chem
from rdkit.Chem import rdFMCS
from utils import sanitize_mol


def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    return new_atom

    
def __merge_molecules(xmol, ymol, mx, my):
    new_mol = Chem.RWMol(xmol)

    atom_map = {} # y_idx -> x_idx
    for atom in ymol.GetAtoms():
        idx = atom.GetIdx()
        if idx in my:
            atom_map[idx] = mx[my.index(idx)]
        else:
            atom_map[idx] = new_mol.AddAtom( copy_atom(atom) )

    for bond in ymol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        bt = bond.GetBondType()
        a1, a2 = atom_map[a1.GetIdx()], atom_map[a2.GetIdx()]
        if new_mol.GetBondBetweenAtoms(a1, a2) is None:
            new_mol.AddBond(a1, a2, bt)

    return new_mol.GetMol()
    
    
def merge_rationales(x, y):
    xmol = Chem.MolFromSmiles(x)
    ymol = Chem.MolFromSmiles(y)

    mcs = Chem.rdFMCS.FindMCS([xmol, ymol], ringMatchesRingOnly=True, completeRingsOnly=True, timeout=1)
    if mcs.numAtoms == 0: return []
    
    mcs = Chem.MolFromSmarts(mcs.smartsString)
    xmatch = xmol.GetSubstructMatches(mcs, uniquify=False)
    ymatch = ymol.GetSubstructMatches(mcs, uniquify=False)
    joined = [__merge_molecules(xmol, ymol, mx, my) for mx in xmatch for my in ymatch]
    joined = [Chem.MolToSmiles(new_mol) for new_mol in joined if sanitize_mol(new_mol)]
    
    return list(set(joined))


if __name__ == "__main__":
    
    import rdkit
    import argparse
    
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL) # mute warnings

    parser = argparse.ArgumentParser()
    parser.add_argument('--rationale1', default="./data/jnk3_rationale.csv")
    parser.add_argument('--rationale2', default="./data/gsk3b_rationale.csv")
    parser.add_argument('--output', default="./data/merged_rationales.csv")
    args = parser.parse_args()

    with open(args.rationale1) as f:
        next(f) # skip header
        rationale1 = set(line.split(",")[1] for line in f)

    with open(args.rationale2) as f:
        next(f) # skip header
        rationale2 = set(line.split(",")[1] for line in f)

    with open(args.output, "w") as f:
        f.write("rationale1,rationale2,merged\n")

        for r1 in rationale1:
            for r2 in rationale2:
                merged_list = merge_rationales(r1, r2)

                for merged in merged_list:
                    f.write("%s,%s,%s\n" %(r1, r2, merged))