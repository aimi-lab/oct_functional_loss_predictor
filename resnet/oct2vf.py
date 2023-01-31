from multiprocessing.sharedctypes import Value
import torch
import torch.nn as nn
import torch.optim as optim
from libs.data_retriever import OCTDataset, Resize
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedGroupKFold
from pathlib import Path
import libs.utils as u
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import pandas as pd
# from random import randint
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


class ONHMaculaModel(nn.Module):

    def __init__(self) -> None:
        super(ONHMaculaModel, self).__init__()

        self.model_onh = getattr(resnet, 'resnet50')(pretrained=True, num_classes=1)
        # state = torch.load(Path(
        #     '/storage/homefs/ds21n601/perimetry_project/resnet/runs/REGR_PRETRAIN_AUGMENT_20220730-172544__ep100_bs032_lr1.00E-02_MD_ONH_INTERFC_SGD_INCREASED61WITH49/regressor_bestR2.pth'
        # ))
        # state = {k.replace('module.', ''): v for k, v in state.items()} # it was enclosed in nn.DataParallel
        # self.model_onh.load_state_dict(state)  
        self.model_onh.fc_final = nn.Identity()      

        # for param in self.model_onh.parameters():
        #     param.requires_grad_(False)

        self.model_thickmaps = getattr(resnet, 'resnet50')(pretrained=True, num_classes=1)
        # state = torch.load(Path(
        #     '/storage/homefs/ds21n601/perimetry_project/resnet/runs/REGR_PRETRAIN_AUGMENT_20220730-172344__ep100_bs032_lr1.00E-02_MD_THICK_INTERFC_SGD_INCREASED61WITH49/regressor_bestR2.pth'
        # ))
        # state = {k.replace('module.', ''): v for k, v in state.items()} # it was enclosed in nn.DataParallel
        # self.model_thickmaps.load_state_dict(state)      
        self.model_thickmaps.fc_final = nn.Identity()      

        # for param in self.model_thickmaps.parameters():
        #     param.requires_grad_(False)

        self.drop_out = nn.Dropout(p=0.5)
        self.shrinker = nn.Linear(512 * 8, 64)
        # self.leaky_relu = nn.LeakyReLU()
        self.regressor = nn.Linear(64, 1)

    def forward(self, thick_img, onh_img):

        pred_onh = self.model_onh(onh_img.clone())
        pred_thick = self.model_thickmaps(thick_img.clone())

        out = torch.cat([pred_onh, pred_thick], dim=1)
        out = self.drop_out(out)
        out = self.shrinker(out)
        # out = self.leaky_relu(out)
        out = self.drop_out(out)
        out = self.regressor(out)
        return out


class OCT2VFRegressor:

    def __init__(self, args) -> None:

        self.data_path = DIR_UBELIX if args.ubelix else DIR_LOCAL
        self.slices_path = self.data_path.joinpath(DIR_SLICES)

        self.current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        self._num_classes = 1 if args.target == 'MD' else 10
        self.args = args
        os.makedirs(Path(__file__).parent.joinpath("weights"), exist_ok=True)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

        if args.command == 'infer': return

        path_str = "REGR_PRETRAIN_AUGMENT_{}__ep{:02d}_bs{:03d}_lr{:.2E}_{}_{}_{}_{}_IMPROVE_RT".format(
        # path_str = "REGR_PRETRAIN_AUGMENT_{}__ep{:02d}_bs{:03d}_lr{:.2E}_{}_{}_SGD_INCREASED61WITH49_GRAYIMGS_FROZEN-MORE_FLIP-OD_FUSED_NO-LEAKY_WEIGHTED".format(
            self.current_time,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.target.replace(' ', '-'),
            args.images.upper(),
            'ADAM' if self.args.adam else 'SGD',
            args.model_name.upper()
        )

        self.tb_path = Path(__file__).resolve().parents[0].joinpath("runs", path_str)
        self.writer = SummaryWriter(self.tb_path)

        with open(self.tb_path / 'commandline_args.json', 'w') as f:
            json.dump(args.__dict__, f, indent=2)


    def load_datasets(self):

        if self.args.resize:
            t = transforms.Compose([
                Resize(self.args.resize),
                transforms.ToTensor(),  
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]) 
        else:
            t = transforms.Compose([
                transforms.ToTensor(),  
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        trainvalset = OCTDataset('crossval.csv', transform_image=t, thick_or_onh=self.args.images, target=self.args.target)
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

        testset = OCTDataset('test.csv', transform_image=t, thick_or_onh=self.args.images, target=self.args.target)
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

    def load_model(self, weights_from=None):

        if self.args.images in ['onh', 'thick']:
            model = getattr(resnet, self.args.model_name)(pretrained=True, num_classes=self._num_classes)

            # if self.args.model_name in ['resnet34']: # ['resnet18', 'resnet34']:
                # model.layer3 = nn.BatchNorm2d(128 if self.args.model_name in ['resnet18', 'resnet34'] else 512)
            # # model.layer3 = nn.Identity()
                # model.layer4 = nn.Identity()
            #     layers = [model.layer2, model.layer3, model.avgpool, model.fc_final]
            # else:
                # layers = [model.layer4, model.avgpool, model.fc_final]
            layers = [model.layer4]
            # model.fc_final = nn.Linear(128 if self.args.model_name in ['resnet18', 'resnet34'] else 2048, self._num_classes)
            model.fc_final = nn.Linear(512 if self.args.model_name in ['resnet18', 'resnet34'] else 2048, self._num_classes)

        else:
            model = ONHMaculaModel()
            layers = [model.model_onh.layer4, model.model_thickmaps.layer4, model.model_onh.avgpool, model.model_thickmaps.avgpool, model.regressor, model.shrinker]

        # for param in model.parameters():
        #     param.requires_grad = False

        # for layer in layers:
        #     for param in layer.parameters():
        #         param.requires_grad = True

        if weights_from is not None:
            assert isinstance(weights_from, Path)
            print('Loading weights from already-trained model')
            state = torch.load(weights_from)
            state = {k.replace('module.', ''): v for k, v in state.items()} # if it was enclosed in nn.DataParallel
            model.load_state_dict(state)

        print(f'GPU devices: {torch.cuda.device_count()}')
        self.model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        self.model.to(self.device)

    def compute_loss(self, outputs, values, criterion, weights):

        if self.args.weighted and self._num_classes > 1:
            raise NotImplementedError('Cannot performed weighted loss calculation on more than one class')

        if self._num_classes == 1:
            if self.args.weighted:
                loss = criterion(
                            torch.mul(outputs, weights.view(-1, 1)),
                            torch.mul(values.view(-1, 1), weights.view(-1, 1))
                        )
            else:
                loss = criterion(outputs, values.view(-1, 1)) 
        else:
            loss = 0
            for ii in range(self._num_classes):
                cluster_loss = criterion(outputs[:, ii], values[:, ii])
                loss += cluster_loss
        return loss

    def train(self):
        self.model.train()

        if self.args.adam:
            optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        else:
            optimizer = optim.SGD(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-6, momentum=0.9)
        lmbda = lambda epoch: 0.99
        scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')#, min_lr=self.args.learning_rate/10)

        criterion = nn.L1Loss() 
        # criterion = nn.MSELoss() 

        best_r2 = 0
        
        for epoch in range(self.args.epochs):

            for phase in ['train', 'test', 'validation']:
                running_loss = 0.0
                running_pred = []
                running_true = []

                if phase == 'train':
                    self.model.train()
                    loader = self.trainloader
                    OCTDataset.augment_image = True
                    writing_freq = self.writing_freq_train
                    i_train = 0
                elif phase == 'validation':
                    self.model.eval()
                    loader = self.valloader
                    OCTDataset.augment_image = False
                    writing_freq = self.writing_freq_val
                elif phase == 'test':
                    self.model.eval()
                    loader = self.testloader
                    OCTDataset.augment_image = False
                    writing_freq = self.writing_freq_test

                for i, data in enumerate(loader):
                    values = data['values'].to(self.device)
                    weights = data['weights'].to(self.device)

                    with torch.set_grad_enabled(phase == 'train'):

                        if self.args.images == 'thick':
                            inputs_thick = data['images_thick'].to(self.device).float()
                            outputs = self.model(inputs_thick)
                        elif self.args.images == 'onh':
                            inputs_onh = data['images_onh'].to(self.device).float()
                            outputs = self.model(inputs_onh)
                        else:
                            assert self.args.images == 'combined'
                            inputs_thick = data['images_thick'].to(self.device).float()
                            inputs_onh = data['images_onh'].to(self.device).float()
                            outputs = self.model(inputs_thick, inputs_onh)

                        # implement custom loss with sample weights computation
                        loss = self.compute_loss(outputs, values, criterion, weights)

                        if phase == 'train':
                            i_train = i
                            loss.backward()
                            optimizer.step()
                            optimizer.zero_grad()

                    running_loss += loss.item()

                    running_pred.append(outputs.detach().cpu().numpy())
                    running_true.append(values.detach().cpu().numpy())

                    if i % writing_freq == (writing_freq - 1):

                        n_epoch = epoch * self.trainset_size // self.args.batch_size + i_train + 1
                        epoch_loss = running_loss / (writing_freq * self.args.batch_size)
                        dict_metrics = u.calculate_metrics(running_true, running_pred)
                        epoch_mae = dict_metrics['MAE']
                        epoch_r2 = dict_metrics['R2']
                        # epoch_rocauc_weighted = dict_metrics['ROC AUC weighted']
                        print(f'{phase} Loss: {epoch_loss} MAE: {epoch_mae}')
                        dict_metrics['Loss'] = epoch_loss
                        dict_metrics['LR'] = optimizer.param_groups[0]["lr"]

                        # TODO: add image to summary writer
                        u.write_to_tb(self.writer, dict_metrics.keys(), dict_metrics.values(), n_epoch, phase=phase)

                        if phase == 'train' and not self.args.images == 'combined':                        
                            # img_idx = randint(0, 10)
                            self.writer.add_figure('train example', loader.dataset.dataset.get_sample(0), global_step=n_epoch)
                        if phase == 'test':
                            self.writer.add_figure('test true preds', u.plot_truth_prediction(running_true, running_pred), global_step=n_epoch)

                        running_pred = []
                        running_true = []
                        running_loss = 0.0

                        if phase == 'validation' and epoch_r2 > best_r2: 
                            best_r2 = epoch_r2
                            best_model = copy.deepcopy(self.model) 

            # pass validation MAE/R2 to LR scheduler
            # scheduler.step(epoch_r2)
            scheduler.step(epoch_r2)
            print(f'Epoch {epoch + 1} finished')
                
            # torch.save(self.model.state_dict(),
            #         Path(__file__).parents[0].joinpath('weights', f'detector_{self.current_time}_e{epoch + 1}.pth'))
    
        # Save best models and create symlink in working directories
        best_r2_model = Path(__file__).parents[0].joinpath(
            'weights', f'regressor_{self.current_time}_bestR2.pth'
        )
        torch.save(best_model.state_dict(), best_r2_model)
        self.tb_path.joinpath('regressor_bestR2.pth').symlink_to(best_r2_model)

        self.infer(best_model, self.tb_path, gradcam=self.args.grad_cam)

    def infer(self, model, save_dir, gradcam=False):

        save_dir.mkdir(exist_ok=True)
        OCTDataset.augment_image = False
        u.eval_model(model, self.testloader, self.device, save_dir, self.args.images, dtype='test')
        u.eval_model(model, self.valloader, self.device, save_dir, self.args.images, dtype='validation')
        u.eval_model(model, self.trainloader, self.device, save_dir, self.args.images, dtype='train')

        if gradcam:
            gradcam_dir = save_dir.joinpath('gradcam')
            gradcam_dir.mkdir(exist_ok=True)
            u.make_output_images(model, self.testloader, self.device, gradcam_dir, self.args.images, self._num_classes)

        # GradCAM
        # csv_paths_gradcam = [data_path.joinpath(f'annotation_30_percent_export.csv')]
        # gradcamset = OCTSlicesDataset('test', csv_paths_gradcam, slices_path, TARGET, transform_image=t_test)
        # testloader = torch.utils.data.DataLoader(gradcamset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        # pathh = Path(FLAGS.model_path) if FLAGS.model_path else tb_path
        # with open(pathh / 'output_thresholds.json') as json_file:
        #     threshold_json = json.load(json_file)
        # for k, v in threshold_json.items():
        #     threshold_json[k] = float(v)
            


if __name__ == '__main__':
    ONHMaculaModel()
    print('ciao')