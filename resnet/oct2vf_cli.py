import argparse
from oct2vf import OCT2VFRegressor
# from pathlib import Path

def run_training(args):
    regr = OCT2VFRegressor(args)
    regr.load_datasets()
    regr.load_model()
    regr.train()

parser = argparse.ArgumentParser(
    prog='OCT2VF Glaucoma MD Regressor', 
    epilog="See '<command> --help' to read about a specific sub-command.",
    fromfile_prefix_chars='@'
)

parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')

# Common arguments to train and infer subparsers
# parser = argparse.ArgumentParser(add_help=False)                                 
# parent_parser.add_argument('--test-data', action="store", type=int, required=True)
# parent_parser.add_argument('--classes', action="store", type=int, required=True)
parser.add_argument('--resize', action="store", type=int)
parser.add_argument('--target', action="store", type=str, required=True)
parser.add_argument('--model-name', action="store", type=str, required=True)
parser.add_argument('--no-ubelix', action="store_false", dest="ubelix")
parser.add_argument("--batch-size", action="store", type=int, required=True)
parser.add_argument("--epochs", action="store", type=int, required=True)
parser.add_argument("--learning-rate", action="store", type=float, required=True)
parser.add_argument("--writing-per-epoch", action="store", type=int, default=10)

# subparsers = parser.add_subparsers(dest='command', help='Sub-commands')

# train_parser = subparsers.add_parser('train', help='Train a model', parents=[parent_parser])
# train_parser.add_argument("--ensemble", action="store_true")
# train_parser.add_argument("--model-paths", action="extend", nargs="+", type=str)
# train_parser.add_argument('--train-data', action="store", type=int, required=True)
# train_parser.add_argument('--train-target', action="store", type=str, required=True)
# train_parser.add_argument('--no-pretrain', action="store_false", dest="pretrained")
# train_parser.add_argument('--restart-from', action="store", type=str)
# train_parser.set_defaults(func=run_command)

# infer_parser = subparsers.add_parser('infer', help='Run inference with a trained model', parents=[parent_parser])
# infer_parser.add_argument('--model-path', default="store", type=str, required=True)
# infer_parser.add_argument("--batch-size", action="store", type=int, default=32)
# infer_parser.add_argument('--baseline', action='store_true')
# infer_parser.add_argument('--no-baseline', dest='baseline', action='store_false')
# infer_parser.set_defaults(baseline=True)
# infer_parser.set_defaults(func=run_command)

args = parser.parse_args()

run_training(args)