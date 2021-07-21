from my_model import Classifier
from data_io import ASVspoof2019TrillMeanDataModule
from tensorneko import Trainer
from run_utils import *
from metrics.eval_metrics import *
import torch
import wandb

hyperparameter_defaults = dict(
    mlp_neurons=1024,
    layers=2,
    dropout_rate=0.5,
    activation_function="LeakyReLU"
)

wandb.init(config=hyperparameter_defaults)
config = wandb.config

hidden_layers = config.layers
hidden_neurons = config.mlp_neurons
dropout_rate: float = config.dropout_rate
activation_function: str = config.activation_function

batch_size = 128
num_worker = 1

model = Classifier(
     hidden_layers, 
     hidden_neurons, 
     activation_function, 
     dropout_rate
)

dm = ASVspoof2019TrillMeanDataModule(batch_size=batch_size, num_workers=num_worker)
dm.setup()

trainer = Trainer(logger=None, checkpoint_callback=False, gpus=1, log_every_n_steps=0, max_epochs=200)

trainer.fit(model, dm)

model.eval()

outs = []
labels = []

for x, y in tqdm(dm.val_dataloader()):
    outs.append(model(x))
    labels.append(y)
    
out = torch.cat(outs).cpu().detach()
true = torch.cat(labels).cpu().detach()
pred = out.max(dim=1)[1]

corr = true == pred

target_scores = []
nontarget_scores = []

for i in range(len(true)):
    if corr[i]:
        target_scores.append(out[i, 1])
    else:
        nontarget_scores.append(out[i, 1])

target_scores = torch.Tensor(target_scores).numpy()
nontarget_scores = torch.Tensor(nontarget_scores).numpy()

eer = compute_eer(target_scores, nontarget_scores)[0]

metrics = {
    "eer": eer
}

wandb.log(metrics)