import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from pathlib import Path
import torch
from grappa.models.energy import Energy
from grappa.models.grappa import GrappaModel
import os
from typing import Union


def get_lightning_trainer(max_epochs=500, gradient_clip_val=1e1, profiler="simple", early_stopping_criterion='early_stopping_loss', config={}, name:str=None, notes:str=None, project='grappa', resume_id:str=None, wandb_dir:Union[str,Path]=Path.cwd(), **kwargs)->pl.Trainer:
    """
    Returns a pytorch lightning trainer with a wandb logger.
    Initializes wandb.
    """

    # Generate a unique ID for the run
    # run_id = wandb.util.generate_id()

    # Initialize Wandb
    if resume_id is None:
        wandb.init(project=project, notes=notes, dir=str(wandb_dir))
        # get the run name:
        if name is not None:
            wandb.run.name += f'-{name}'

    else:
        # check whether the run is running:
        api = wandb.Api()
        run = api.run(f"{project}/{resume_id}")
        if run.state == 'running':
            raise ValueError(f"Run {resume_id} is still running. Please wait until it is finished.")
        
        wandb.init(id=resume_id, resume=True, project=project, notes=notes)



    # Initialize WandbLogger with the existing run
    wandb_logger = WandbLogger(experiment=wandb.run, notes=notes, project=project, name=wandb.run.name)

    # Checkpoint directory in the run directory
    # print(dir(wandb_logger.experiment))
    wandb_run_dir = wandb_logger.experiment.dir
    assert isinstance(wandb_run_dir, str) or isinstance(wandb_run_dir, Path), f"Internal Error, found type {type(wandb_run_dir)}"
    checkpoint_dir = str(Path(wandb_run_dir) / "checkpoints")

    print('checkpoint_dir:', checkpoint_dir)


    ###############################
    # CHECKPOINTING MODIFICATIONS:
    # SAVE MODEL WITHOUT ENERGY MODULE
    # WRITE CONFIG DICT IN THE CHECKPOINT


    class CustomModelCheckpoint(pl.callbacks.ModelCheckpoint):
        def __init__(self, config, **kwargs):
            super().__init__(**kwargs)
            self.config = config
            self.make_initial_checkpoint = not resume_id is None

        # def on_train_start(self, trainer, pl_module):
        #     # Save a checkpoint before the first epoch begins
        #     if self.make_initial_checkpoint:
        #         self.make_initial_checkpoint = False
        #         self.on_epoch_end(trainer, pl_module)

        def on_epoch_end(self, trainer, pl_module):
            torch.save({
                'state_dict': pl_module.state_dict(),
                'config': self.config,
            }, self.filepath.format(epoch=trainer.current_epoch, **trainer.logged_metrics))

            super().on_epoch_end(trainer, pl_module)


    # ModelCheckpoint callback with dynamic path and filename
    checkpoint_callback = CustomModelCheckpoint(
        monitor=early_stopping_criterion,
        dirpath=checkpoint_dir,
        filename='best-model',
        save_top_k=1,
        mode='min',
        save_last=True,
        every_n_epochs=5,
        config=config
    )

    # deactivate the progress bar in slurm jobs to prevent huge log files
    is_slurm = os.environ.get('SLURM_JOB_ID') is not None

    if is_slurm:
        print("Detected SLURM mode: Disabling progress bar...")

    trainer = pl.Trainer(logger=wandb_logger, gradient_clip_val=gradient_clip_val, max_epochs=max_epochs, profiler=profiler, callbacks=[checkpoint_callback], enable_progress_bar=not is_slurm, **kwargs)

    return trainer


# own exception class for when a run has failed:
class pl_RunFailed(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)