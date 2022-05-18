import pandas as pd
from sys import path
from sys import platform
import argparse
import os

parser = argparse.ArgumentParser(description="Run decision model")
parser.add_argument(
    "--model",
    choices=["biased", "heuristic", "rw", "rwdecay", "rwrl"],
    type=str,
    help="Craving model to run",
    required=True,
)
parser.add_argument("--rerun", default=False, action="store_true")
parser.add_argument(
    "--cores", type=int, help="Number of cores to use for multiprocessing", default=4
)
parser.add_argument(
    "--draws", type=int, help="Number of samples to draw from posterior", default=1000
)
args = parser.parse_args()

root_dir = os.path.abspath('../')
## Add model_functions to system path
path.append(root_dir)

from utils import load_data, plotting
from models import RescorlaWagner, Biased, Heuristic, RWRL, RWDecay

if __name__ == "__main__":

    path_to_summary = f"{root_dir}/sample_data/clean_df_summary.csv"
    path_to_longform = f"{root_dir}/sample_data/clean_df_longform.csv"
    df_summary, longform = load_data.load_clean_dbs(path_to_summary, path_to_longform)
    netcdf_path = f"{root_dir}/decision_output/"

    if args.model == "biased":
        model = Biased.Biased(
            model_name=args.model,
            save_path=netcdf_path,
            summary=df_summary,
            longform=longform,
        )

    elif args.model == "heuristic":
        model = Heuristic.Heuristic(
            model_name=args.model,
            save_path=netcdf_path,
            summary=df_summary,
            longform=longform,
        )

    elif args.model == "rw":
        model = RescorlaWagner.RW(
            model_name=args.model,
            save_path=netcdf_path,
            summary=df_summary,
            longform=longform,
        )

    elif args.model == "rwdecay":
        model = RWDecay.RWDecay(
            model_name=args.model,
            save_path=netcdf_path,
            summary=df_summary,
            longform=longform,
        )

    elif args.model == "rwrl":
        model = RWRL.RWRL(
            model_name=args.model,
            save_path=netcdf_path,
            summary=df_summary,
            longform=longform,
        )

    model.fit(jupyter=False, cores=args.cores, rerun=args.rerun, draws=args.draws)
    model.calc_Q_table()
    model.calc_bics()

