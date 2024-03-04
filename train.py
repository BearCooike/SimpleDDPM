
import os

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, Flowers102, CelebA, StanfordCars, Food101
from torchvision.utils import save_image

from models.diffuser.gauss import GaussDiffuser
from models.backbone.unet import GhostUNet, UNet

from torch.cuda import amp
from copy import deepcopy

modelConfig = {
    'time_steps': 1000,
    'depth': 4,
    'in_channels': 3,
    'out_channels': 3,
    'dims': [128, 128, 256, 512, 512],
    'num_blocks': [2, 2, 2, 2, 2],
    'attn': [False, False, True, False],
}
trainConfig = {
    'dataset': 'cifar10',
    'weight': './logs/exp7/best.pt',
    'weight': None,
    'batch_size': 8,
    'img_size': (128,128),
    'lr0':1e-5,
    'epochs': 300,
    "beta_1": 1e-4,
    "beta_T": 0.02,
    'save_weight_dir': './logs/exp1'
}

def train():
    device = torch.device('cuda:0')
    if os.path.exists(trainConfig["save_weight_dir"]) is False:
        os.makedirs(trainConfig["save_weight_dir"])
    # dataset
    if trainConfig['dataset'] == 'cifar10':
        dataset = CIFAR10(
            root='./CIFAR10', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize(trainConfig['img_size']),
                transforms.ToTensor(),
            ]))
    elif trainConfig['dataset'] == 'Flowers102':
        dataset = Flowers102(
            root='./Flowers102', split='train', download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize(trainConfig['img_size']),
                transforms.ToTensor(),
            ]))
    elif trainConfig['dataset'] == 'Food101':
        dataset = Food101(
            root='./Food101', split='train', download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize(trainConfig['img_size']),
                transforms.ToTensor(),
            ]))
    
    dataloader = DataLoader(
        dataset, batch_size=trainConfig["batch_size"], shuffle=True, num_workers=16, drop_last=True, pin_memory=True)

    # model setup
    scaler = amp.GradScaler(enabled=True)

    net_model = UNet(**modelConfig).to(device)
    if trainConfig["weight"] is not None:
        net_model.load_state_dict(torch.load(trainConfig["weight"], map_location=device)['model'])
    optimizer = torch.optim.AdamW(net_model.parameters(), lr=trainConfig["lr0"], weight_decay=1e-4)
    # optimizer = torch.optim.SGD(net_model.parameters(), lr=trainConfig["lr0"], weight_decay=1e-4, momentum=0.9, nesterov=True)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=trainConfig["epochs"], eta_min=0, last_epoch=-1)

    trainer = GaussDiffuser(
        net_model, trainConfig["beta_1"], trainConfig["beta_T"], modelConfig["time_steps"]).to(device)

    min_loss = 99999

    trainer.train()
    # start training
    for e in range(trainConfig["epochs"]):
        mloss = torch.zeros(1, device=device)
        with tqdm(enumerate(dataloader), dynamic_ncols=True, total=len(dataloader)) as tqdmDataLoader:
            for i, (images, labels) in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                x_0 = images.float().to(device)
                with amp.autocast(enabled=True):
                    loss = trainer(x_0) / modelConfig["time_steps"]
                    # loss = trainer(x_0)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), 1)

                scaler.step(optimizer)  # optimizer.step
                scaler.update()

                mloss = (mloss * i + loss.detach()) / (i + 1)

                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": f"{e}/{trainConfig['epochs']}",
                    "loss": mloss.item()/trainConfig["batch_size"],
                    "img shape": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        cosineScheduler.step()
        if mloss.item() < min_loss:
            min_loss = mloss.item()
            ckpt = {}
            ckpt['model'] = deepcopy(net_model).half().state_dict()
            ckpt['config'] = modelConfig
            ckpt['img_size'] = trainConfig['img_size']
            torch.save(ckpt, os.path.join(
                trainConfig["save_weight_dir"], 'best' + ".pt"))
        
if __name__ == '__main__':
    train()