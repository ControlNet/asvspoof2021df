from data_io import ASVspoof2019TrillMeanDataModule0, ASVspoof2019TrillMeanDataModule1
from my_model import AnomalyDetector
from tensorneko import NekoTrainer
import torch
import wandb
import numpy as np
from tqdm import tqdm

hyperparameter_defaults = dict(
    encoder_layers=3,
    latent_neurons=1024,
    dropout_rate=0.5,
    activation_function="LeakyReLU",
    noise_sigma=0.1
)

wandb.init(config=hyperparameter_defaults)
config = wandb.config

encoder_layers = config.encoder_layers
latent_neurons = config.latent_neurons
dropout_rate: float = config.dropout_rate
activation_function: str = config.activation_function
noise_sigma = config.noise_sigma

batch_size = 128
num_worker = 10

model = AnomalyDetector(
    encoder_layers,
    latent_neurons,
    activation_function,
    dropout_rate,
    noise_sigma
)

dm1 = ASVspoof2019TrillMeanDataModule1(batch_size=batch_size, num_workers=num_worker)
dm1.setup()
dm0 = ASVspoof2019TrillMeanDataModule0(batch_size=batch_size, num_workers=num_worker)
dm0.setup()

trainer_ae = NekoTrainer(logger=None, checkpoint_callback=False, gpus=1, log_every_n_steps=0, max_epochs=500)

trainer_ae.fit(model.ae, dm1)

dm1_loss = model.ae.history[-1]["val_loss"].item()
min_loss = min(list(map(lambda x: x["val_loss"] if "val_loss" in x else torch.tensor([100]), model.ae.history))).item()

dm0_loss = []


cuda = torch.device("cuda")
model = model.to(cuda)

for batch in tqdm(dm0.val_dataset):
    x, y = batch
    x = x.to(cuda)
    x_pred = model.ae(x)
    loss = model.ae.criterion(x_pred, x)
    dm0_loss.append(loss.item())

metrics = {
    "loss": dm1_loss,
    "min_loss": min_loss,
    "dm0_loss": np.mean(dm0_loss)
}

wandb.log(metrics)
