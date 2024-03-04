import os
import torch
from models.backbone.unet import GhostUNet, UNet
from models.diffuser.gauss import GaussDiffuser
from torchvision.utils import save_image

torch.backends.cudnn.benchmark = True

sampleConfig = {
    # 'weight': './logs/exp2/ckpt_30_.pt',
    'weight': './logs/exp7/best.pt',
    'mode': 'ddpm',
    'batch_size': 4,
    'lr0':1e-4,
    'epochs': 100,
    "beta_1": 1e-4,
    "beta_T": 0.02,
    'save_sample_dir': './samples/exp7/'
}

def eval():
    if os.path.exists(sampleConfig["save_sample_dir"]) is False:
        os.makedirs(sampleConfig["save_sample_dir"])
    # load model and evaluate
    with torch.no_grad():
        device = torch.device('cuda:0')
        ckpt = torch.load(sampleConfig['weight'], map_location=device)
        modelConfig = ckpt['config']
        img_size = ckpt['img_size']
        print(modelConfig)
        model = UNet(**modelConfig).to(device)
        model.load_state_dict(ckpt['model'])
        print("model load weight done.")
        # model.half()
        model.eval()
        sampler = GaussDiffuser(
            model, sampleConfig["beta_1"], sampleConfig["beta_T"], modelConfig["time_steps"]).to(device)
        sampler.eval()
        
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[sampleConfig["batch_size"], 3, *img_size], device=device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        noisy_name = os.path.join(sampleConfig['save_sample_dir'], 'noisy.png')
        save_image(saveNoisy, noisy_name, nrow=4)
        sampledImgs = sampler(noisyImage, mod=sampleConfig['mode'])
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        sample_name = os.path.join(sampleConfig['save_sample_dir'], 'sample.png')
        save_image(sampledImgs, sample_name, nrow=4)

if __name__ == "__main__":
    eval()