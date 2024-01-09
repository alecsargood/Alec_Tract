#!/usr/bin/env python

import argparse
import comet_ml  # noqa: F401 ugh
import torch

from argparse import RawTextHelpFormatter
from comet_ml import Experiment as CometExperiment

from TrackToLearn.algorithms.NFsac_auto import NFSACAuto
from TrackToLearn.experiment.experiment import (
    add_data_args,
    add_environment_args,
    add_experiment_args,
    add_model_args,
    add_tracking_args)
from TrackToLearn.experiment.train import (
    add_rl_args,
    TrackToLearnTraining)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert torch.cuda.is_available()


class NFSACAutoTrackToLearnTraining(TrackToLearnTraining):
    """
    Main RL tracking experiment
    """

    def __init__(
        self,
        nfsac_auto_train_dto: dict,
        comet_experiment: CometExperiment,
    ):
        """
        Parameters
        ----------
        sac_auto_train_dto: dict
            SACAuto training parameters
        comet_experiment: CometExperiment
            Allows for logging and experiment management.
        """

        super().__init__(
            nfsac_auto_train_dto,
            comet_experiment,
        )

        # SACAuto-specific parameters
        self.alpha = nfsac_auto_train_dto['alpha']
        self.num_flows = nfsac_auto_train_dto['num_flows']

    def save_hyperparameters(self):
        """ Add SACAuto-specific hyperparameters to self.hyperparameters
        then save to file.
        """

        self.hyperparameters.update(
            {'algorithm': 'NFSACAuto',
             'alpha': self.alpha,
             'num_flows': self.num_flows})
        super().save_hyperparameters()

    def get_alg(self, max_nb_steps: int):
        alg = NFSACAuto(
            self.input_size,
            self.action_size,
            self.hidden_dims,
            self.num_flows,
            self.lr,
            self.gamma,
            self.alpha,
            self.n_actor,
            self.rng,
            device)
        return alg


def add_nfsac_auto_args(parser):
    parser.add_argument('--alpha', default=0.2, type=float,
                        help='Temperature parameter')
    parser.add_argument('--num_flows', default=4, type=int,
                        help='Number of Flows')


def parse_args():
    """ Generate a tractogram from a trained recurrent model. """
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=RawTextHelpFormatter)

    add_experiment_args(parser)
    add_data_args(parser)

    add_environment_args(parser)
    add_model_args(parser)
    add_rl_args(parser)
    add_tracking_args(parser)

    add_nfsac_auto_args(parser)

    arguments = parser.parse_args()
    return arguments


def main():
    """ Main tracking script """
    args = parse_args()
    print(args)

    experiment = CometExperiment(project_name=args.experiment,
                                 workspace='alecsargood',
                                 api_key='eQ6pfPreHQzFB4frzYlCeLiEr',
                                 parse_args=False,
                                 auto_metric_logging=False,
                                 disabled=not args.use_comet)

    nfsac_auto_experiment = NFSACAutoTrackToLearnTraining(
        # Dataset params
        vars(args),
        experiment
    )
    nfsac_auto_experiment.run()


if __name__ == '__main__':
    main()
