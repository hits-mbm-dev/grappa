    
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, default="grappa-1.2", help="Project name for wandb.")
parser.add_argument("-tb", "--train_batch", type=int, default=-1, help="Batch size for training.")
parser.add_argument("-vb", "--val_batch", type=int, default=-1, help="Batch size for validation.")
# parser.add_argument("--pretrain_path", type=str, default=None, help="Path to pretrained model.") #NOTE: include this arg
parser.add_argument("-p", "--param_weight", type=float, default=None, help="Weight for the param loss of the datasets with classical parameters from amber99sbildn. Default is None.")
parser.add_argument("-b", "--bondbreak_radicals", action="store_true", default=False, help="Whether to include bond breaking radicals in the training set. Default is False.")
parser.add_argument("--shrink_train", type=float, default=None, help="Subsample factor for the training set (default: None).")
parser.add_argument("--n_periodicity", type=int, default=3, help="Number of periodicity for the torsion features.")
parser.add_argument("--no_torsion_cutoff", action="store_true", help="Do not use the torsion cutoff.")
parser.add_argument("--pretrain_path", type=str, default=None, help="Path to pretrained model used for initialization.")
parser.add_argument("--train_only_am1bcc", action="store_true", help="Train only on am1bcc datasets. This can be used for evaluating the difference in charge models.")

if __name__ == "__main__":

    args = parser.parse_args()


    from grappa.training.trainrun import do_trainrun
    import yaml
    from pathlib import Path
    import numpy as np

    # load the config:
    config_path = "grappa_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # set the splitpath:    
    config["data_config"]["splitpath"] = 'espaloma_split'

    # set the name:
    config["trainer_config"]["name"] = ""

    # set the batch sizes:
    if args.train_batch > 0:
        config["data_config"]["train_batch_size"] = args.train_batch
    if args.val_batch > 0:
        config["data_config"]["val_batch_size"] = args.val_batch

    if not args.param_weight is None:
        config["trainer_config"]["name"] += f"_p{int(np.log10(args.param_weight))}"
        config['lit_model_config']['param_weights_by_dataset'] = {ds: args.param_weight for ds in config['data_config']['datasets'] if 'amber99sbildn' in ds}

    if args.bondbreak_radicals:
        config["trainer_config"]["name"] += "_bondbreak"
        config['data_config']['datasets'].append('AA_bondbreak_rad_amber99sbildn')
        # increase the number of occurence of the bond breaking radicals since there are only few molecules in that subdataset:
        config['data_config']['weights']['AA_bondbreak_rad_amber99sbildn'] = 2.


    if args.shrink_train is not None:
        config["data_config"]["tr_subsampling_factor"] = args.shrink_train
        config["trainer_config"]["name"] += f"_shrink{int(args.shrink_train*100)}"


    if args.n_periodicity != 3:
        config["model_config"]["n_periodicity_proper"] = int(args.n_periodicity)
        config["trainer_config"]["name"] += f"_n_p{int(args.n_periodicity)}"


    if args.no_torsion_cutoff:
        config["model_config"]["torsion_cutoff"] = 0.
        config["trainer_config"]["name"] += "_no_cutoff"


    if args.pretrain_path is not None:
        # set the param loss epochs to 0
        config['lit_model_config']['param_loss_epochs'] = 0
        config['trainer_config']['name'] += "_pretrain"

    if args.train_only_am1bcc:
        config['data_config']['pure_val_datasets'] = [ds for ds in config['data_config']['datasets'] if 'amber99sbildn' in ds or "_rad" in ds]
        config['data_config']['datasets'] = [ds for ds in config['data_config']['datasets'] if not 'amber99sbildn' in ds and not "_rad" in ds]
        config['trainer_config']['name'] += "_am1bcc"
        config['model_config']['in_feat_name'].remove('charge_model')
        
    # train:
    do_trainrun(config=config, project=args.project, pretrain_path=args.pretrain_path, dir=Path(__file__).parent)