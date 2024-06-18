import zipfile
import os
import requests
from tqdm import tqdm
from pathlib import Path
from typing import Union
from tqdm import tqdm

def get_repo_dir()->Path:
    '''
    Returns the path to the root of the repository.
    '''
    return Path(__file__).parents[3]

def get_data_path()->Path:
    '''
    Returns the default path where to look for datasets.
    '''
    return get_repo_dir() / "data"


# NOTE: this is currently only used for the split loading... maybe delete the method.
def get_path_from_tag(tag:str, data_dir:Union[Path,str]=get_data_path()/'dgl_datasets')->Path:
    '''
    Returns the path to a dataset given a tag. If the dataset is not at the corresponding location, it is downloaded. The tag is the dirname of the dataset, available tags are:

    SPLITFILES:
        'espaloma_split'
    '''

    dir_path = Path(data_dir) / tag

    if dir_path.exists():
        return dir_path
    
    # else, construct the dgl dataset from a folder with moldata files, thus, return a moldata path
    moldata_path = get_moldata_path(tag)
    return moldata_path


def get_moldata_path(tag:str, data_dir:Union[Path,str]=get_data_path()/'datasets')->Path:
    '''
    Returns the path to a dataset given a tag. If the dataset is not at the corresponding location, it is downloaded. The tag is the dirname of the dataset, available tags are:

    BENCHMARK ESPALOMA:
        - 'spice-des-monomers'
        - 'spice-pubchem'
        - 'gen2'
        - 'gen2-torsion'
        - 'spice-dipeptide'
        - 'protein-torsion'
        - 'pepconf-dlc'
        - 'rna-diverse'
        - 'rna-trinucleotide'

    PEPTIDE DATASET:
        - dipeptides-300K-openff-1.2.0
        - dipeptides-300K-amber99
        - dipeptides-300K-charmm36
        - dipeptides-1000K-openff-1.2.0
        - dipeptides-1000K-amber99
        - dipeptides-1000K-charmm36
        - uncapped-300K-openff-1.2.0
        - uncapped-300K-amber99
        - dipeptides-hyp-dop-300K-amber99

    RADICAL DATASET:
        - dipeptides-radical-300K
        - bondbreak-radical-peptides-300K

    SPLITFILE:
        'espaloma_split'
    '''

    RELEASE_URL = 'https://github.com/hits-mbm-dev/grappa/releases/download/v.1.2.0/'

    URL_TAGS = [
        'spice-des-monomers',
        'spice-pubchem',
        'gen2',
        'gen2-torsion',
        'rna-diverse',
        'rna-trinucleotide',
        'rna-nucleoside',
        'spice-dipeptide',
        'protein-torsion',
        'pepconf-dlc',
        'spice-dipeptide_amber99sbildn',
        'tripeptides_amber99sbildn',
        'dipeptide_rad',
        'hyp-dop_amber99sbildn',
        'uncapped_amber99sbildn',
        'AA_bondbreak_rad_amber99sbildn',
        'espaloma_split',
    ]

    tag = str(tag)

    if (Path(data_dir)/tag).exists():
        if not list((Path(data_dir)/tag).iterdir()):
            raise RuntimeError(f'The dataset path {Path(data_dir)/tag} exists but is empty. Please remove the directory.')
        return Path(data_dir)/tag

    # Download the files and put them in the dir if it doesn't exist
    if not (str(tag) in URL_TAGS):
        raise ValueError(f"Tag {tag} not recognized. Available tags for download are {URL_TAGS}")
    
    url = RELEASE_URL + tag + '.zip'

    return load_dataset(url=url, data_dir=data_dir, dirname=tag)



def load_dataset(url:str, data_dir:Path=get_data_path()/'datasets', dirname:str=None)->Path:
    """
    Downloads a zip dataset from a given URL if it's not already present in the local directory, 
    then extracts it.

    Parameters:
        url (str): The URL of the dataset to download.
        data_dir (str): The local directory to store and extract the dataset. Default is 'grappa/data/dgl_datasets'.

    Returns:
        str: Path to the directory where the dataset is extracted.
    """

    data_dir = Path(data_dir).absolute()

    # Create the directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)

    # Extract dirname from URL
    if dirname is None:
        dirname = url.split('/')[-1].split('.')[0]
    dir_path = data_dir / dirname


    # Download the file if it doesn't exist
    if not dir_path.exists():
        print(f"Downloading {dirname} from:\n'{url}'")

        # this is the path to the zip file that is deleted after extraction
        zip_path = dir_path.with_suffix('.zip')

        # Start the download
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Ensure the request was successful

        # Get the total file size from headers
        total_size = int(response.headers.get('content-length', 0))

        # Initialize the progress bar
        with tqdm(total=total_size, unit='B', unit_scale=True) as t:
            with open(zip_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
                    t.update(len(chunk))

        # print(f"Downloaded {zip_path}")

        # Unzip the file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(str(data_dir))
            print(f"Stored dataset at:\n{dir_path}")
        
        # delete the zip file
        os.remove(zip_path)

    return dir_path
