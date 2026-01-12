import argparse

class ArgParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Process argument provided by command line.", add_help=True
        )
        self.args = None
        self._add_required_arguments()
        self._add_optional_arguments()

    def _add_required_arguments(self):
        parser = self.parser
        parser.add_argument(
            "--config",
            type=str,
            help="Path to config file.",
        )

    def _add_optional_arguments(self):
        parser = self.parser

        parser.add_argument(
            "-lr", type=float, help="Learning rate."
        )

        parser.add_argument("-ep", "--max_epochs", type=int, help="Number of epochs.", )

        parser.add_argument("-bs", "--batch_size", type=int, help="Batch size.")

        parser.add_argument("-g", "--gpus", type=int, default=1, help="Number of GPUs the trainer can use. Default is 1.")

    def parse_args(self):
        if self.args is None:
            self.args = self.parser.parse_args()
        return self.args

    def get_overrides(self):
        """
        Returns overrides in a dot list format which can be read by omega config.
        """
        self.parse_args()
        args = vars(self.args)

        mapping = {
            "batch_size": [
                "dataloader.train.batch_size",
                "dataloader.val.batch_size",
                "dataloader.test.batch_size",
            ],
            "max_epochs": ["trainer.max_epochs"],
            "lr": ["optimizer.lr"],
        }

        overrides = []
        for key, values in mapping.items():
            if args[key]:
                overrides.extend([f'{val}={args[key]}' for val in values])

        return overrides
