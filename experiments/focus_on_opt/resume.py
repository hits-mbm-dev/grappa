from grappa.training.resume_trainrun import resume_trainrun
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("run_id", type=str, help="Run id of the run to resume.")
parser.add_argument("--project", type=str, default="grappa-1.0", help="Project name for wandb.")

args = parser.parse_args()

resume_trainrun(run_id=args.run_id, project=args.project, new_wandb_run=False, overwrite_config={'lit_model_config': {'time_limit': 1000}, 'data_config': {'datasets':['spice-des-monomers']}})