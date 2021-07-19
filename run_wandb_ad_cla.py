from data_io import ASVspoof2019TrillMean, ASVspoof2019TrillMeanDataModule, ASVspoof2019TrillMeanDataModule0, ASVspoof2019Trill
from my_model import AnomalyDetector
from pytorch_lightning.callbacks import ModelCheckpoint
from tensorneko import NekoTrainer
from tqdm import tqdm
from run_utils import *
from metrics.eval_metrics import * 

from torch.utils.data import DataLoader
from tensorneko.util import summarize_dict_by

import numpy as np
import torch
import wandb

hyperparameter_defaults = dict(
    cla_hidden_layers = 1,
    cla_hidden_neurons = 256,
    cla_activation = "GELU",
    cla_dropout_rate = 0.3,
    cla_attn = False,
    cla_basis = "linear",
    cla_norm = False
)
wandb.init(config=hyperparameter_defaults)
config = wandb.config


encoder_layers = 1
latent_neurons = 2048
dropout_rate: float = 0.05
activation_function: str = "LeakyReLU"
noise_sigma = 0.001625
    
cla_hidden_layers = config.cla_hidden_layers
cla_hidden_neurons = config.cla_hidden_neurons
cla_activation = config.cla_activation
cla_dropout_rate = config.cla_dropout_rate
cla_attn = config.cla_attn
cla_basis = config.cla_basis
cla_norm = config.cla_norm

batch_size = 128
num_worker = 10

dm = ASVspoof2019TrillMeanDataModule(batch_size=batch_size, num_workers=num_worker)
dm.setup()


model = AnomalyDetector(
    encoder_layers,
    latent_neurons,
    activation_function,
    dropout_rate,
    noise_sigma,
    cla_hidden_layers, 
    cla_hidden_neurons, 
    cla_activation,
    cla_dropout_rate,
    cla_attn,
    cla_basis,
    cla_norm
)

model.ae = model.ae.load_from_checkpoint("ckpt3/trill_mean_anomaly_detector_ae-epoch=293-val_loss=0.001.ckpt")

trainer = NekoTrainer(logger=None, checkpoint_callback=False, gpus=1, log_every_n_steps=0, max_epochs=500)

trainer.fit(model, dm)

model.eval()


eer_history = list(map(lambda x: x["val_eer"].item(), model.history))
eer = eer_history[-1]
max_eer = max(eer_history)

metrics = {
    "eer": eer,
    "max_eer": max_eer
}

wandb.log(metrics)
