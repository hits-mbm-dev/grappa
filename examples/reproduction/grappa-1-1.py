#%%
from grappa.utils.loading_utils import model_dict_from_tag
from grappa.training.trainrun import do_trainrun
import json
from pathlib import Path
#%%

thisdir = Path(__file__).parent
# published grappa models have not only the state-dict of the trained model, but also the results of the training, the configuration used for training and the partition of the dataset into train, validation and test molecules.

model_dict = model_dict_from_tag('grappa-1.1.0')

# the config dict contains hyperparameters for the model and training and also information about the datasets used
config = model_dict['config']

# create a file in which the names of the train/val/test molecules are stored
split_ids = model_dict['split_names']
splitpath = str(thisdir / 'split_ids.json')
with open(splitpath, 'w') as f:
    json.dump(split_ids, f)

config['data_config']['splitpath'] = splitpath

# this will start a trainrun that is logged in wandb
do_trainrun(config=config, project='reproduce-grappa', dir=thisdir)