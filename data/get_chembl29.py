import os


if __name__ == "__main__":
    # download   
    unzip_path = 'chembl_29_sqlite'
    if not os.path.exists(unzip_path):
        download_path = 'chembl_29_sqlite.tar.gz'
        print('downloading data to %s ...' % download_path)
        source = 'https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_29/chembl_29_sqlite.tar.gz'
        os.system('wget -O %s %s' % (download_path, source))
        print('finished downloading')
        
        # unzip
        print('extracting data to %s ...' % unzip_path)
        os.system('tar zxvf %s' % download_path)
        print('finished extracting')