import os


def read_xyz(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        smiles = lines[-2].split('\t')[0]
    return smiles


if __name__ == "__main__":
    # download   
    unzip_path = 'qm9_raw'
    if not os.path.exists(unzip_path):
        download_path = 'dsgdb9nsd.xyz.tar.bz2'
        print('downloading data to %s ...' % download_path)
        source = 'https://ndownloader.figshare.com/files/3195389'
        os.system('wget -O %s %s' % (download_path, source))
        print('finished downloading')
        
        # unzip
        print('extracting data to %s ...' % unzip_path)
        os.mkdir(unzip_path)
        os.system('tar xvjf %s -C %s' % (download_path, unzip_path))
        print('finished extracting')