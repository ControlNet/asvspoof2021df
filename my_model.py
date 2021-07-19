from typing import Optional, Union, Sequence, Dict

from fn import _
from pytorch_lightning.utilities.types import EPOCH_OUTPUT

from tensorneko import NekoModel, NekoModule
from torch import Tensor
import torch
from torch import nn
from torch.nn import ModuleList
from torch.optim import Adam

from tensorneko.module import MLP
from tensorneko.layer import Linear
from tensorneko.util import get_activation, summarize_dict_by
import torchmetrics
from metrics.eval_metrics import compute_eer


class GaussianNoise(NekoModule):
    def __init__(self, sigma=0.1, device = "cuda"):
        super().__init__()
        self.sigma = sigma
        self.noise = torch.tensor(0.).to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach()
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


class ModelBase(NekoModel):

    def training_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        optimizer_idx: Optional[int] = None, hiddens: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        batch_x, batch_y = batch
        batch_out = self(batch_x)
        batch_loss = self.criterion(batch_out, batch_y)
        batch_acc = self.acc(batch_out.max(dim=1)[1], batch_y)
        return {"loss": batch_loss, "acc": batch_acc, "pred": batch_out, "label": batch_y}

    def validation_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None
    ) -> Dict[str, Tensor]:
        batch_x, batch_y = batch
        batch_out = self(batch_x)
        batch_loss = self.criterion(batch_out, batch_y)
        batch_acc = self.acc(batch_out.max(dim=1)[1], batch_y)
        return {"loss": batch_loss, "acc": batch_acc, "pred": batch_out, "label": batch_y}

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None) -> Tensor:
        batch_x, batch_y = batch
        batch_out = self(batch_x)
        return batch_out

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        super().training_epoch_end(list(map(lambda out: {"loss": out["loss"], "acc": out["acc"]}, outputs)))
        self.eval()

        out = summarize_dict_by("pred", _)(outputs).detach().cpu()
        pred = out.max(dim=1)[1]
        true = torch.cat(list(map(lambda x: x["label"], outputs))).cpu()
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

        self.log("eer", eer, on_epoch=True, on_step=False, logger=True, prog_bar=True)
        if len(self.history) > 0:
            self.history[-1]["eer"] = eer
        self.train()

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        super().validation_epoch_end(list(map(lambda out: {"loss": out["loss"], "acc": out["acc"]}, outputs)))
        self.eval()

        out = summarize_dict_by("pred", _)(outputs).detach().cpu()
        pred = out.max(dim=1)[1]
        true = torch.cat(list(map(lambda x: x["label"], outputs))).cpu()

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
        self.log("eer", eer, on_epoch=True, on_step=False, logger=True, prog_bar=False)
        self.log("val_eer", eer, on_epoch=True, on_step=False, logger=True, prog_bar=True)
        if len(self.history) > 0:
            self.history[-1]["val_eer"] = eer

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=0.0001)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.6,
                    patience=10,
                    min_lr=1e-8,
                    verbose=True
                ),
                "monitor": "val_loss"
            }
        }


class Classifier(ModelBase):

    def __init__(self,
        hidden_layers: int,
        hidden_neurons: int,
        activation: str,
        dropout: float
    ):
        super().__init__("trill_mean_attn_mlp")
        self.save_hyperparameters()
        # set objective (loss) functions
        weight = torch.FloatTensor([0.1, 0.9]).cuda()
        self.criterion = nn.CrossEntropyLoss(weight=weight)
        self.activation = get_activation(activation)

        self.hidden_layers = hidden_layers

        self.mlp = MLP([12288] + [hidden_neurons] * hidden_layers + [2], build_activation=self.activation,
            dropout_rate=dropout
        )

        self.attns = nn.ModuleList([
            Linear(hidden_neurons, hidden_neurons, build_activation=nn.Sigmoid) for i in range(hidden_layers)
        ])

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.acc = torchmetrics.Accuracy()

    def forward(self, x: Tensor) -> Tensor:
        for i in range(self.hidden_layers):
            x = self.mlp.layers[i](x)
            w = self.attns[i](x)
            x = w * x

        x = self.mlp.layers[-1](x)
        x = self.log_softmax(x)
        return x


class AnomalyDetector(ModelBase):

    def __init__(self,
        encoder_layers: int,
        latent_neurons: int,
        activation: str,
        dropout: float,
        noise_sigma: float=0,
        cla_hidden_layers=0,
        cla_hidden_neurons=None,
        cla_activation=None,
        cla_dropout_rate=0,
        cla_attn=False,
        cla_basis="linear",
        cla_norm=False
    ):
        super().__init__("trill_mean_anomaly_detector")
        self.save_hyperparameters()
        # set objective (loss) functions
        weight = torch.FloatTensor([0.1, 0.9]).cuda()
        self.criterion = nn.CrossEntropyLoss(weight=weight)
        self.activation = get_activation(activation)
        self.cla_activation = get_activation(cla_activation) if cla_activation else None

        self.cla_hidden_layers = cla_hidden_layers
        self.ae = AnomalyDetector.AE(encoder_layers, latent_neurons, activation, dropout, noise_sigma)
        self.mlp = MLP([12288] + [cla_hidden_neurons] * cla_hidden_layers + [2], build_activation=self.cla_activation,
            dropout_rate=cla_dropout_rate)
        
        self.cla_attn = cla_attn
        if cla_attn and cla_hidden_layers > 0:
            self.attns = nn.ModuleList(MLP(
                [cla_hidden_neurons] * (cla_hidden_layers + 1), build_activation=nn.Sigmoid
            ).layers)
        else:
            self.attns = None

        self.basis = basis_func_dict()[cla_basis]
        self.norm = nn.BatchNorm1d(12288) if cla_norm else nn.Identity()

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.acc = torchmetrics.Accuracy()

    def forward(self, x: Tensor) -> Tensor:
        x_rec = self.ae.eval()(x)
        x = x_rec - x
        x = self.basis(x)
        x = self.norm(x)
        for i in range(self.cla_hidden_layers):
            x = self.mlp.layers[i](x)
            if self.cla_attn:
                w = self.attns[i](x)
                x = w * x
        x = self.mlp.layers[-1](x)
        x = self.log_softmax(x)
        return x

    def configure_optimizers(self):
        optimizer = Adam(self.mlp.parameters(), lr=1e-4, betas=(0.5, 0.9))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.6,
                    patience=10,
                    min_lr=1e-8,
                    verbose=True
                ),
                "monitor": "val_loss"
            }
        }

    class AE(NekoModel):

        def __init__(self, encoder_layers: int, latent_neurons: int, activation, dropout_rate, noise_sigma):
            super().__init__("trill_mean_anomaly_detector_ae")
            self.save_hyperparameters()
            self.criterion = nn.MSELoss()
            self.activation = get_activation(activation)

            encoder_neurons = [12288] + [
                latent_neurons * 2 ** i for i in reversed(range(encoder_layers))
            ]

            decoder_neurons = list(reversed(encoder_neurons))

            self.encoder = ModuleList(
                MLP(encoder_neurons, build_activation=self.activation, dropout_rate=dropout_rate).layers)
            self.decoder = ModuleList(
                MLP(decoder_neurons, build_activation=self.activation, dropout_rate=dropout_rate).layers)
            
            self.noise = GaussianNoise(noise_sigma) if noise_sigma > 0 else nn.Identity()

        def forward(self, x):
            x = self.noise(x)
            xs = []
            for layer in self.encoder:
                xs.append(x := layer(x))
            for layer in self.decoder:
                x = layer(x + xs.pop())
            return x

        def training_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None,
            batch_idx: Optional[int] = None,
            optimizer_idx: Optional[int] = None, hiddens: Optional[Tensor] = None
        ) -> Dict[str, Tensor]:
            batch_x, _ = batch
            batch_out = self(batch_x)
            batch_loss = self.criterion(batch_out, batch_x)
            return {"loss": batch_loss}

        def validation_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None,
            batch_idx: Optional[int] = None,
            dataloader_idx: Optional[int] = None
        ) -> Dict[str, Tensor]:
            batch_x, _ = batch
            batch_out = self(batch_x)
            batch_loss = self.criterion(batch_out, batch_x)
            return {"loss": batch_loss}

        def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None) -> Tensor:
            batch_x, batch_y = batch
            batch_out = self(batch_x)
            return batch_out

        def configure_optimizers(self):
            optimizer = Adam(self.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=0.00002)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode='min',
                        factor=0.6,
                        patience=10,
                        min_lr=1e-8,
                        verbose=True
                    ),
                    "monitor": "val_loss"
                }
            }


# RBFs

def gaussian(alpha):
    phi = torch.exp(-1 * alpha.pow(2))
    return phi


def linear(alpha):
    phi = alpha
    return phi


def quadratic(alpha):
    phi = alpha.pow(2)
    return phi


def inverse_quadratic(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
    return phi


def multiquadric(alpha):
    phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi


def inverse_multiquadric(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi


def spline(alpha):
    phi = (alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha)))
    return phi


def poisson_one(alpha):
    phi = (alpha - torch.ones_like(alpha)) * torch.exp(-alpha)
    return phi


def poisson_two(alpha):
    phi = ((alpha - 2 * torch.ones_like(alpha)) / 2 * torch.ones_like(alpha)) \
          * alpha * torch.exp(-alpha)
    return phi


def matern32(alpha):
    phi = (torch.ones_like(alpha) + 3 ** 0.5 * alpha) * torch.exp(-3 ** 0.5 * alpha)
    return phi


def matern52(alpha):
    phi = (torch.ones_like(alpha) + 5 ** 0.5 * alpha + (5 / 3) * alpha.pow(2)) * torch.exp(-5 ** 0.5 * alpha)
    return phi


def tanh(alpha):
    phi = torch.tanh(alpha)
    return phi


def basis_func_dict():
    """
    A helper function that returns a dictionary containing each RBF
    """

    bases = {
        'gaussian': gaussian,
        'linear': linear,
        'quadratic': quadratic,
        'inverse quadratic': inverse_quadratic,
        'multiquadric': multiquadric,
        'inverse multiquadric': inverse_multiquadric,
        'spline': spline,
        'poisson one': poisson_one,
        'poisson two': poisson_two,
        'matern32': matern32,
        'matern52': matern52,
        "tanh": tanh
    }
    return bases