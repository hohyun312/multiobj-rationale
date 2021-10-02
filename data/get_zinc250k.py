import os


if __name__ == "__main__":
    unzip_path = 'zinc250k'
    if not os.path.exists(unzip_path):
        os.mkdir(unzip_path)
        download_path = '%s/250k_rndm_zinc_drugs_clean_3.csv' % unzip_path
        print('downloading data to %s ...' % download_path)
        source = 'https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv'
        os.system('wget -O %s %s' % (download_path, source))
        print('finished downloading')