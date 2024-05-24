#%%
import torch
torch.set_float32_matmul_precision('medium')
torch.set_default_dtype(torch.float32)

from grappa.models import Energy, deploy

from grappa.utils.run_utils import write_yaml
from pathlib import Path
import wandb
import sys
from grappa.training.export_model import remove_module_prefix
from grappa.utils import graph_utils, dgl_utils
from typing import List, Dict, Union
from grappa.training.get_dataloaders import get_dataloaders
from grappa.training.lightning_model import LitModel
from grappa.training.lightning_trainer import get_lightning_trainer, pl_RunFailed
from grappa.training.config import default_config
from grappa.utils.graph_utils import get_param_statistics, get_default_statistics
from typing import Callable
from grappa.training.resume_trainrun import resume_trainrun

#%%
def do_trainrun(config:Dict, project:str='grappa', config_from_sweep:Callable=None, manual_sweep_config:Callable=None, pretrain_path:Union[Path,str]=None, dir:Union[Path,str]=Path.cwd()):
    """
    Do a single training run with the given configuration.

    config: dictionary with the configurations. config['trainer_config'] may also contain kwargs for pl.Trainer.

    config_from_sweep: function that takes the wandb.config and returns a dict that can be used to update the grappa config. This can be used for updating several config parameters from one sweep param (e.g. global dropout rates) or for applying transformations to the sweep config (e.g. converting a sweep parameter like the learning rate from linear to log scale). This can be e.g.:
        def config_from_sweep(wandb_config):
            config = {}
            config['lit_model_config'] = {}
            config['lit_model_config']['lr'] = wandb_config.lr

            config['model_config'] = {}
            config['model_config']['param_dropout'] = wandb_config.dropout
            config['model_config']['gnn_conv_dropout'] = wandb_config.dropout

            return config

    manual_sweep_config: function that sets wandb.config parameters specified in the sweep to some manual values defined in the function. This can be used for setting the sweep parameters to some known good starting values. Use eg wandb.config.update({'lr': 0.001}, allow_val_change=True) to set the learning rate to 0.001. In this case, the sweep config must be None.
    pretrain_path: path to a checkpoint that is used to initialize the model weights. This can be a lightning checkpoint or the state dict directly.
    In this case, the splitpath from the pretrained model is used and the start_qm_epoch is set to 0.
    """

    RESTRICT_CONFIG = False

    # check whether all config args are allowed (they are allowed if they are in the default config)
    default_config_ = default_config()
    for k in config.keys():
        if k in ["trainer_config"]:
            continue
        if k not in default_config_:
            raise KeyError(f"Key {k} not an allowed config argument.")
        if isinstance(config[k], dict):
            for kk in config[k].keys():
                if kk not in default_config_[k]:
                    if RESTRICT_CONFIG:
                        raise KeyError(f"Key {k}-{kk} not an allowed config argument.")


    # Get the trainer  and initialize wandb
    trainer = get_lightning_trainer(**config['trainer_config'], config=config, project=project, wandb_dir=dir)

    run_id = wandb.run.id

    experiment_dir = trainer.logger.experiment.dir

    if manual_sweep_config is not None:
        manual_sweep_config()

    # update the config according to the config stored at wandb
    if not config_from_sweep is None:
        # get the hp values that were assigned by wandb sweep:
        wandb_stored_config = wandb.config
        updated_config = config_from_sweep(wandb_stored_config)
        assert len(list(updated_config.keys())) > 0

        # overwrite the config values that correspond to them:
        for k in updated_config.keys():
            if not k in config.keys():
                raise ValueError(f'The sweep config provides a key that is not part of the config: {k}')
            for kk in updated_config[k].keys():
                if not kk in config[k].keys():
                    raise ValueError(f'The sweep config provides a key that is not part of the config: {k}/{kk}')
                config[k][kk] = updated_config[k][kk]


    # write the current config to the run directory.
    write_yaml(config, Path(experiment_dir)/"grappa_config.yaml")

    if pretrain_path is not None:
    # set the splitpath to the splitpath of the pretrained model:
        splitpath = str(Path(pretrain_path).parent.parent/'split.json')
        config['data_config']['splitpath'] = splitpath

    # Get the dataloaders
    tr_loader, val_loader, test_loader = get_dataloaders(**config['data_config'], classical_needed=config['lit_model_config']['log_classical'], in_feat_names=config['model_config']['in_feat_name'], save_splits=Path(experiment_dir)/'split.json')

    param_statistics = get_param_statistics(tr_loader)
    for m in ['mean', 'std']:
        for k, v in param_statistics[m].items():
            if torch.isnan(v).any():
                param_statistics[m][k] = get_default_statistics()[m][k]

    # Get the model
    model = deploy.model_from_config(model_config=config['model_config'], param_statistics=param_statistics)     

    gradient_needed = config['lit_model_config']['gradient_weight'] != 0.
    # add energy calculation
    model = torch.nn.Sequential(
        model,
        Energy(suffix='', gradients=gradient_needed),
        # Energy(suffix='_ref', write_suffix="_classical_ff", gradients=gradient_needed)
    )

    if pretrain_path is not None:
        d = torch.load(str(pretrain_path))
        if 'state_dict' in d.keys():
            state_dict = d['state_dict']
            # state_dict = remove_module_prefix(state_dict)
        else:
            state_dict = d

        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            try:
                state_dict = remove_module_prefix(state_dict)
                model.load_state_dict(state_dict)
            except Exception as e2:
                raise e2 from e
            
        # set the start_qm_epoch to 0:
        config['lit_model_config']['start_qm_epochs'] = 0



    model.train()

    # test whether the model can be applied to some input of the train set:
    if config['test_model']:
        test_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        example, _ = next(iter(val_loader))
        example = example.to(test_device)
        model = model.to(test_device)
        example = model(example)
        example_ = dgl_utils.unbatch(example)[0]
        energies_example = graph_utils.get_energies(example_)
        assert not torch.isnan(energies_example).any(), "Model predicts NaN energies."
        assert not torch.isinf(energies_example).any(), "Model predicts inf energies."
        assert len(energies_example) > 0, "Model predicts no energies."

    # Get a pytorch lightning model
    lit_model = LitModel(model=model, tr_loader=tr_loader, vl_loader=val_loader, te_loader=test_loader, **config['lit_model_config'])

    if pretrain_path is not None:
        # set the warmup step to 0 to begin the training with warmup:
        lit_model.warmup_step = 0

    print(f"\nStarting training run {wandb.run.name}...\nSaving logs to {run_id}...\n")

    try:
        # Train the model
        trainer.fit(model=lit_model, train_dataloaders=tr_loader, val_dataloaders=val_loader)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        custom_e = pl_RunFailed(f"Run {run_id} failed with exception {e}.")
        custom_e._run_id = run_id
        raise custom_e

    return model


def safe_trainrun(config:Dict, project:str='grappa', config_from_sweep:Callable=None, manual_sweep_config:Callable=None, pretrain_path:Union[Path,str]=None):
    """
    Like the above but exceptions are caught and the run is restarted (with new run id) from the last checkpoint. This is done only once.
    """
    
    try:
        do_trainrun(config=config, project=project, config_from_sweep=config_from_sweep, manual_sweep_config=manual_sweep_config, pretrain_path=pretrain_path)
    except pl_RunFailed as e:
        print(f"Exception {e} occured. Restarting training run...", file=sys.stderr)
        run_id = e._run_id
        try:
            resume_trainrun(run_id=run_id, project=project, wandb_folder=Path.cwd()/'wandb', new_wandb_run=True, overwrite_config=None)
        except Exception as e2:
            print(f"Exception {e2} occured. Restarting training run failed.", file=sys.stderr)
            raise e2 from e