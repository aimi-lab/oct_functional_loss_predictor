from sklearn import ensemble
import torch
import torch.nn as nn
import torch.optim as optim
from libs.data_retriever import OCTDataset, Resize
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from pathlib import Path
import libs.utils as u
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import pandas as pd
from random import randint
from time import sleep
import os
import copy
import json
from libs import resnet

RNDM = 91
torch.manual_seed(RNDM)
DIR_UBELIX = Path(__file__).parent.parent.joinpath("inputs")
DIR_LOCAL = Path("/storage/homefs/ds21n601/perimetry_project/inputs")
DIR_SLICES = "slices"


class OCT2VFRegressor:

    def __init__(self, args) -> None:
        
        self.data_path = DIR_UBELIX if args.ubelix else DIR_LOCAL
        self.slices_path = self.data_path.joinpath(DIR_SLICES)

        self.current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        os.makedirs('weights', exist_ok=True)
        path_str = "REGR_{}__ep{:02d}_bs{:03d}_lr{:.2E}_{}".format(
            self.current_time,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.target
        )

        self.tb_path = Path(__file__).resolve().parents[0].joinpath("runs", path_str)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
        self.writer = SummaryWriter(self.tb_path)

        with open(self.tb_path / 'commandline_args.json', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        self.args = args

    def load_datasets(self):

        if self.args.resize:
            t = transforms.Compose([
                Resize(self.args.resize),
                transforms.ToTensor(),  
                # transforms.RandomHorizontalFlip(), 
                # transforms.RandomRotation(10),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]) 
            # t_test = transforms.Compose([
            #     Resize(self.args.resize), 
            #     transforms.ToTensor(), 
            #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            # ]) 
        else:
            t = transforms.Compose([
                # transforms.RandomHorizontalFlip(), 
                # transforms.RandomRotation(10),
                transforms.ToTensor(),  
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            # t_test = transforms.Compose([
            #     transforms.ToTensor(), 
            #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            # ]) 

        trainvalset = OCTDataset('crossval.csv', transform_image=t)
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RNDM)
        trainset_indices, valset_indices = list(sgkf.split(trainvalset.index_set, trainvalset.gs_set, groups=trainvalset.patient_set))[0]
        valset_size = len(valset_indices)
        self.trainset_size = len(trainset_indices)
        self.writing_freq_train = self.trainset_size // (self.args.writing_per_epoch * self.args.batch_size)
        self.writing_freq_val = valset_size // self.args.batch_size  # Only once per epoch

        trainset = torch.utils.data.Subset(trainvalset, trainset_indices)
        valset = torch.utils.data.Subset(trainvalset, valset_indices)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True, num_workers=0)
        self.valloader = torch.utils.data.DataLoader(valset, batch_size=self.args.batch_size, shuffle=True, num_workers=0)

        testset = OCTDataset('test.csv', transform_image=t)
        testset_size = len(testset)
        self.writing_freq_test = testset_size // self.args.batch_size  # Only once per epoch
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.args.batch_size, shuffle=True, num_workers=0)

        # with open(self.tb_path / 'validation_indices.csv', 'w') as fp:
        #     fp.write('\n'.join([str(ii) for ii in valset_indices]))

        # with open(self.tb_path / 'test_indices.csv', 'w') as fp:
        #     fp.write('\n'.join([str(ii) for ii in testset_indices]))

        print(self.trainset_size, valset_size, testset_size)

        # pd.DataFrame(
        #     zip(testvalset.df.iloc[testset_indices].sum(), testvalset.df.iloc[valset_indices].sum()), 
        #     columns=['test', 'val'], 
        #     index=testvalset.df.columns
        # ).to_csv(self.tb_path / 'val_test_distribution.csv')

    def load_model(self):

        model = getattr(resnet, self.args.model_name)(pretrained=getattr(self.args, 'pretrained', False), num_classes=1)
        
        print(f'GPU devices: {torch.cuda.device_count()}')
        self.model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        self.model.to(self.device)

    def train(self):
        self.model.train()

        optimizer = optim.SGD(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-6, momentum=0.9)
        lmbda = lambda epoch: 0.99
        scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

        # print(f'Applied weights for training loss will be: {self.trainloader.dataset.weights.numpy()}')

        criterion = nn.L1Loss() 

        best_r2 = 0
        
        for epoch in range(self.args.epochs):

            for phase in ['train', 'test', 'validation']:
                running_loss = 0.0
                running_pred = []
                running_true = []

                if phase == 'train':
                    self.model.train()
                    loader = self.trainloader
                    writing_freq = self.writing_freq_train
                    i_train = 0
                elif phase == 'validation':
                    self.model.eval()
                    loader = self.valloader
                    writing_freq = self.writing_freq_val
                elif phase == 'test':
                    self.model.eval()
                    loader = self.testloader
                    writing_freq = self.writing_freq_test

                for i, data in enumerate(loader):
                    inputs, values = data['images'].to(self.device).float(), data['values'].to(self.device)

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        # FIXME: need to scale data?
                        loss = criterion(outputs, values.view(-1,1)) # sigmoid is included in BCEWithLogitsLoss

                        if phase == 'train':
                            i_train = i
                            loss.backward()
                            optimizer.step()
                            optimizer.zero_grad()

                    running_loss += loss.item()

                    # FIXME: removed sigmoid
                    running_pred.append(outputs.detach().cpu().numpy())
                    running_true.append(values.detach().cpu().numpy())

                    if i % writing_freq == (writing_freq - 1):

                        n_epoch = epoch * self.trainset_size // self.args.batch_size + i_train + 1
                        epoch_loss = running_loss / (writing_freq * self.args.batch_size)
                        dict_metrics = u.calculate_metrics(running_true, running_pred)
                        epoch_mae = dict_metrics['MAE']
                        epoch_r2 = dict_metrics['R2']
                        # epoch_rocauc_weighted = dict_metrics['ROC AUC weighted']
                        print(f'{phase} Loss: {epoch_loss} ROC AUC: {epoch_mae}')
                        dict_metrics['Loss'] = epoch_loss
                        u.write_to_tb(self.writer, dict_metrics.keys(), dict_metrics.values(), n_epoch, phase=phase)

                        if phase == 'train':                        
                            self.writer.add_figure('train example', loader.get_sample(0), global_step=n_epoch)
                        elif phase == 'test':
                            self.writer.add_figure('test true preds', u.plot_truth_prediction(running_true, running_pred), global_step=n_epoch)

                        running_pred = []
                        running_true = []
                        running_loss = 0.0

                        if phase == 'validation' and epoch_r2 > best_r2: 
                            best_r2 = epoch_r2
                            best_model = copy.deepcopy(self.model) 

            scheduler.step()
            print(f'Epoch {epoch + 1} finished')
                
            # torch.save(self.model.state_dict(),
            #         Path(__file__).parents[0].joinpath('weights', f'detector_{self.current_time}_e{epoch + 1}.pth'))
    
        # Save best models and create symlink in working directories
        best_rocauc_model_path = Path(__file__).parents[0].joinpath(
            'weights', f'regressor_{self.current_time}_bestR2.pth'
        )
        torch.save(best_model.state_dict(), best_rocauc_model_path)
        self.tb_path.joinpath('regressor_bestR2.pth').symlink_to(best_rocauc_model_path)

        # FIXME: implement final inference plots
        # self.infer(best_model, self.tb_path)

    def infer(self, model, save_dir, gradcam=False):

        save_dir.mkdir(exist_ok=True)

        # assert (self.testloader.dataset.dataset.df.columns == self.trainloader.dataset.df.columns).all()
        assert (self.testloader.dataset.dataset.df.columns == self.valloader.dataset.dataset.df.columns).all()

        u.eval_model(model, self.valloader, self.testloader, self.device, save_dir)

        # GradCAM
        # csv_paths_gradcam = [data_path.joinpath(f'annotation_30_percent_export.csv')]
        # gradcamset = OCTSlicesDataset('test', csv_paths_gradcam, slices_path, TARGET, transform_image=t_test)
        # testloader = torch.utils.data.DataLoader(gradcamset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        # pathh = Path(FLAGS.model_path) if FLAGS.model_path else tb_path
        # with open(pathh / 'output_thresholds.json') as json_file:
        #     threshold_json = json.load(json_file)
        # for k, v in threshold_json.items():
        #     threshold_json[k] = float(v)
            
        # u.make_output_images(best_model, testloader, device, tb_path, data_path.joinpath(f'annotation_30_percent_export.csv'), threshold_json)

