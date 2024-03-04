# Simple DDPM

A simple DDPM implementation

## train

```bash
$ python train.py
```

You can change the training configuration by modifying `modelConfig` and `trainConfig`

```python
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
    'weight': None,
    'batch_size': 8,
    'img_size': (32,32),
    'lr0':1e-5,
    'epochs': 300,
    "beta_1": 1e-4,
    "beta_T": 0.02,
    'save_weight_dir': './logs/exp1'
}
```

## sample

For an already trained model, we can use `sample.py` to generate it.

```bash
$ python sample.py
```

```python
sampleConfig = {
    'weight': './logs/exp1/best.pt',
    'mode': 'ddpm',
    'batch_size': 4,
    'lr0':1e-4,
    'epochs': 100,
    "beta_1": 1e-4,
    "beta_T": 0.02,
    'save_sample_dir': './samples/exp1/'
}
```

