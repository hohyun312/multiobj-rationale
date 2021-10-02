import os

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