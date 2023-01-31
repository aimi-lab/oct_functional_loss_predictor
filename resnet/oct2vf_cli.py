import argparse
import pathlib
import json
import weakref
from oct2vf import OCT2VFRegressor
# from pathlib import Path

def run_training(args):
    regr = OCT2VFRegressor(args)
    regr.load_datasets()
    regr.load_model()
    regr.train()

def run_inference(args):
    model_dir = pathlib.Path(args.model_dir)
    with open(model_dir.joinpath('commandline_args.json'), 'r') as f:
        json_dict = json.load(f)

    for key in json_dict.keys():
        if not hasattr(args, key):
            setattr(args, key, json_dict[key])

    if hasattr(args, 'images') and args.images == 'combined':
        raise NotImplementedError('Inference on combined images is not implemented')

    model_weights = model_dir.joinpath('regressor_bestR2.pth')
    inference_dir = model_dir.joinpath('inference_dir')

    regr = OCT2VFRegressor(args)
    regr.load_datasets()
    regr.load_model(weights_from=model_weights)
    regr.infer(regr.model, model_dir, gradcam=args.grad_cam) #changed from inference_dir


parser = argparse.ArgumentParser(
    prog='OCT2VF Glaucoma MD Regressor', 
    epilog="See '<command> --help' to read about a specific sub-command.",
    fromfile_prefix_chars='@'
)

parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')

# Common arguments to train and infer subparsers
# parser = argparse.ArgumentParser(add_help=False)                                 
# parent_parser = argparse.ArgumentParser(add_help=False)

subparsers = parser.add_subparsers(dest='command', help='Sub-commands')

train_parser = subparsers.add_parser('train', help='Train a model') #, parents=[parent_parser])
train_parser.add_argument('--resize', action="store", type=int)
train_parser.add_argument('--target', action="store", type=str, required=True, choices=['MD', 'clusters'])
train_parser.add_argument('--model-name', action="store", type=str, required=True)
train_parser.add_argument('--no-ubelix', action="store_false", dest="ubelix")
train_parser.add_argument("--batch-size", action="store", type=int, required=True)
train_parser.add_argument("--epochs", action="store", type=int, required=True)
train_parser.add_argument("--learning-rate", action="store", type=float, required=True)
train_parser.add_argument("--writing-per-epoch", action="store", type=int, default=10)
train_parser.add_argument("--images", action="store", type=str, choices=['onh', 'thick', 'combined'], default='thick')
train_parser.add_argument('--grad-cam', action="store_true")
train_parser.add_argument('--weighted', action="store_true")
train_parser.add_argument('--adam', action="store_true")

infer_parser = subparsers.add_parser('infer', help='Infer on a model') #, parents=[parent_parser])
infer_parser.add_argument('--model-dir', action="store", type=str, required=True)
infer_parser.add_argument('--grad-cam', action="store_true")

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

if args.command == 'train':
    run_training(args)
else:
    assert args.command == 'infer'
    run_inference(args)