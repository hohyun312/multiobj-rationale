import os


def select_jnk3_gskb(download_dir):
        
    jnk3 = []
    gsk3b = []

    with open(os.path.join(download_dir, "pubchem.chembl.dataset4publication_inchi_smiles.tsv"), 'r') as f:
        next(f)
        while True:

            line = f.readline()

            if line: 
                data = line.rstrip().split('\t')
                activity_flag, gene_symbol, smiles = data[3], data[8], data[11]

                if activity_flag == 'A':
                    activity_flag = '1'
                elif activity_flag == 'N':
                    activity_flag = '0'

                if gene_symbol == 'MAPK10':
                    jnk3.append([smiles, activity_flag])

                elif gene_symbol == 'GSK3B':
                    gsk3b.append([smiles, activity_flag])

            else: 
                break

    with open(os.path.join(download_dir, "jnk3.csv"), 'w') as f:
        f.write('smiles,active\n')
        f.write('\n'.join((','.join(j) for j in jnk3)))

    with open(os.path.join(download_dir, "gsk3b.csv"), 'w') as f:
        f.write('smiles,active\n')
        f.write('\n'.join((','.join(g) for g in gsk3b)))
        
        
        
if __name__ == "__main__":
    # download   
    unzip_path = 'excape-db'
    if not os.path.exists(unzip_path):
        os.mkdir(unzip_path)
        download_path = '%s/pubchem.chembl.dataset4publication_inchi_smiles.tsv.xz' % unzip_path
        print('downloading data to %s ...' % download_path)
        source = 'https://zenodo.org/record/173258/files/pubchem.chembl.dataset4publication_inchi_smiles.tsv.xz'
        os.system('wget -O %s %s' % (download_path, source))
        print('finished downloading')
        
        # unzip
        print('extracting data ...')
        os.system('xz -d %s' % download_path)
        print('finished extracting')
    
    print('selecting jnk3 gsk3b data from excape db ...')
    select_jnk3_gskb(unzip_path)
    print('file saved: jnk3.csv, gsk3b.csv')
        
