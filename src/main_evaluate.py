import os
import argparse

import torch

from utils.general_utils import maybe_download_file

from environment.env import CReM_Env

from evaluate.eval_dgaqn import eval_dgaqn
from evaluate.eval_greedy import eval_greedy

def molecule_arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    add_arg = parser.add_argument

    # EXPERIMENT PARAMETERS
    add_arg('--data_path', required=True)
    add_arg('--warm_start_dataset', required=True)
    add_arg('--artifact_path', required=True)
    add_arg('--name', default='default_run')
    add_arg('--use_cpu', action='store_true')
    add_arg('--gpu', default='0')

    add_arg('--greedy', action='store_true')
    add_arg('--model_path', default='')

    add_arg('--reward_type', type=str, default='plogp', help='plogp;logp;dock')

    add_arg('--nb_sample_crem', type=int, default=128)

    add_arg('--nb_test', type=int, default=50)
    add_arg('--nb_bad_steps', type=int, default=5)

    # AUTODOCK PARAMETERS
    add_arg('--obabel_path', default='')
    add_arg('--adt_path', default='')
    add_arg('--receptor_file', default='')

    add_arg('--adt_tmp_dir', default='')

    return parser

def load_dgaqn(model_path):
    dgaqn_model = torch.load(model_path, map_location='cpu')
    print("DGAQN model loaded")
    return dgaqn_model

def main():
    args = molecule_arg_parser().parse_args()
    print("====args====\n", args)

    env = CReM_Env(args.data_path,
                args.warm_start_dataset,
                nb_sample_crem = args.nb_sample_crem,
                mode='mol')

    artifact_path = os.path.join(args.artifact_path, args.name)
    os.makedirs(artifact_path, exist_ok=True)

    if args.greedy is True:
        # Greedy
        eval_greedy(artifact_path,
                    env,
                    args.reward_type,
                    N = args.nb_test,
                    K = args.nb_bad_steps,
                    args = args)
    else:
        # DGAQN
        model = load_dgaqn(args.model_path)
        print(model)
        eval_dgaqn(artifact_path,
                    model,
                    env,
                    args.reward_type,
                    N = args.nb_test,
                    K = args.nb_bad_steps,
                    args = args)


if __name__ == '__main__':
    main()