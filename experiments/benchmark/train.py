if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="benchmark-grappa-1.0", help="Project name for wandb.")
    parser.add_argument("-tb", "--train_batch", type=int, default=-1, help="Batch size for training.")
    parser.add_argument("-vb", "--val_batch", type=int, default=-1, help="Batch size for validation.")
    parser.add_argument("--with_hybridization", action="store_true", help="Use hybridization as input feature. Default is False.")
    parser.add_argument("-o", "--opt-weight", type=float, default=1., help="Sampling factor for the opt-datasets (gen2/pepconf-dlc). Default: 1.0")
    parser.add_argument("-s", "--scan-weight", type=float, default=1., help="Sampling factor for the torsion-scan-datasets (gen2-torsion/protein-torsion). Default: 1.0")
    parser.add_argument("--pretrain_path", type=str, default=None)
    args = parser.parse_args()


    from grappa.training.trainrun import do_trainrun
    import yaml
    from pathlib import Path

    # load the config:
    config_path = "grappa_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # set the splitpath:    
    config["data_config"]["splitpath"] = str(Path(__file__).parent.parent.parent/f"dataset_creation/get_espaloma_split/espaloma_split.json")

    # set the name:
    config["trainer_config"]["name"] = ""

    # set the batch sizes:
    if args.train_batch > 0:
        config["data_config"]["train_batch_size"] = args.train_batch
    if args.val_batch > 0:
        config["data_config"]["val_batch_size"] = args.val_batch

    if args.with_hybridization:
        config["model_config"]["in_feat_name"] += ["sp_hybridization"]
        config["trainer_config"]["name"] += "_hybrid"

    if args.opt_weight != 1.:
        config["trainer_config"]["name"] += f"_opt{int(args.opt_weight)}"
        for d in ["pepconf-dlc", "gen2"]:
            config["data_config"]["weights"][d] = args.opt_weight

    if args.scan_weight != 1.:
        config["trainer_config"]["name"] += f"_scan{int(args.scan_weight)}"
        for d in ["gen2-torsion", "protein-torsion"]:
            config["data_config"]["weights"][d] = args.scan_weight

    if args.pretrain_path is not None:
        config["lit_model_config"]['start_qm_epochs'] = 0

    # train:
    do_trainrun(config=config, project=args.project, pretrain_path=args.pretrain_path)

