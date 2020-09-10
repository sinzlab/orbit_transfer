import torch
from torch import nn
import torch.utils.data as Data

import matplotlib.pyplot as plt

from tqdm import tqdm
import copy

from bias_transfer.gp.nn_kernel import compute_cov_matrix
from bias_transfer.gp.utils import plot_nn

torch.manual_seed(1)  # reproducible


class Sin(torch.nn.Module):
    def forward(self, x, inverse=False):
        if not inverse:
            return torch.sin(x)
        else:
            return torch.asin(x)


class InvNet(torch.nn.Module):
    def __init__(
        self,
        input_size,
        num_layers,
        layer_size,
        output_size,
        activation="sigmoid",
        dropout=0.0,
    ):
        super(InvNet, self).__init__()

        if activation == "sin":
            activation_module = Sin()
        elif activation == "tanh":
            activation_module = torch.nn.Tanh()
        elif activation == "relu":
            activation_module = torch.nn.ReLU()
        else:
            activation_module = torch.nn.Sigmoid()

        self.layers = nn.ModuleList([nn.Linear(input_size, layer_size, bias=True)])
        self.layers.append(activation_module)
        for i in range(1, num_layers - 1):
            if dropout:
                self.layers.append(nn.Dropout(p=dropout))
            self.layers.append(nn.Linear(layer_size, layer_size, bias=True))
            self.layers.append(activation_module)
        if dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.layers.append(nn.Linear(layer_size, output_size, bias=False))

    def forward(self, x, inverse=False):
        if not inverse:
            for i, layer in enumerate(self.layers):
                x = layer(x)
        else:
            for i, layer in enumerate(self.layers[::-1]):
                if isinstance(layer, nn.Linear):
                    x = torch.inverse(layer.weight.detach()) @ x
                else:
                    x = layer(x, inverse=True)
        return x

    def __getitem__(self, val):
        if isinstance(val, slice):
            clone = copy.deepcopy(self)
            clone.layers = clone.layers[val]
            return clone
        else:
            return self.layers[val]


def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


def freeze_core(net):
    net.load_state_dict(torch.load("best_model.pth"))
    params = net.parameters()
    for param in params:
        param.requires_grad = False


def get_net(activation="sigmoid", dropout=0.0, width=100, layers=4):
    net = InvNet(
        input_size=1,
        num_layers=layers,
        layer_size=width,
        output_size=1,
        activation=activation,
        dropout=dropout,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(device)
    return net, device


def trainer(
    net,
    X_train,
    Y_train,
    rdm_matching=False,
    device="cpu",
    epoch=400,
    batch_size=32,
    optim="Adam",
    scheduler=False,
):
    if optim == "Adam":
        optimizer = torch.optim.Adam(
            net.parameters(), lr=0.0001, amsgrad=False, weight_decay=1e-6
        )
    else:
        optimizer = torch.optim.SGD(
            net.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.9
        )
    if scheduler:
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer, lr_lambda=[lambda epoch: (1 + 0.0001 * epoch) ** (-0.25)]
        # )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[50,100,200,400], gamma=0.5)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

    # if rdm_matching:
    #     torch_dataset = Data.TensorDataset(torch.tensor(X_plot,dtype=torch.float),torch.tensor(X_plot,dtype=torch.float))
    #     torch_dataset = Data.TensorDataset(torch.tensor(X_plot,dtype=torch.float),torch.tensor(X_plot,dtype=torch.float))
    # else:
    train_len = X_train.shape[0]
    dev_start = int(train_len * 0.9)
    train = Data.TensorDataset(
        torch.tensor(X_train[:], dtype=torch.float),
        torch.tensor(Y_train[:], dtype=torch.float),
    )
    # dev = Data.TensorDataset(
    #     torch.tensor(X_train[dev_start:], dtype=torch.float),
    #     torch.tensor(Y_train[dev_start:], dtype=torch.float),
    # )

    train_loader = Data.DataLoader(
        dataset=train, batch_size=batch_size, shuffle=False, num_workers=1,
    )
    # dev_loader = Data.DataLoader(
    #     dataset=dev, batch_size=batch_size, shuffle=True, num_workers=2,
    # )

    # start training
    if hasattr(
        tqdm, "_instances"
    ):  # To have tqdm output without line-breaks between steps
        tqdm._instances.clear()
    t = tqdm(range(epoch))
    min_loss = 1000000
    best_state = None
    dev_step = 1
    # epoch_dev_loss = 0.0
    for epoch in t:
        epoch_loss = 0.0
        net.train()
        for step, (batch_x, batch_y) in enumerate(
            train_loader
        ):  # for each training step

            b_x = batch_x.to(device)
            b_y = batch_y.to(device)

            prediction = net(b_x)  # input x and predict based on x

            if rdm_matching:
                rdm = compute_cov_matrix(prediction, prediction)
                kernel_matrix = torch.tensor(
                    fitted_kernel(b_x.cpu().numpy(), b_x.cpu().numpy()),
                    dtype=torch.float,
                ).to(device)
                #             loss = loss_func(torch.exp(rdm.reshape((-1))), torch.exp(kernel_matrix.reshape((-1))))    # must be (1. nn output, 2. target)
                loss = loss_func(
                    rdm.reshape((-1)), kernel_matrix.reshape((-1))
                )  # must be (1. nn output, 2. target)
            else:
                loss = loss_func(
                    prediction.reshape((-1)), b_y
                )  # must be (1. nn output, 2. target)

            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            epoch_loss += loss.item()
            t.set_postfix(
                epoch_loss=epoch_loss / (step + 1),
                # epoch_dev_loss=epoch_dev_loss / (dev_step + 1),
            )
        net.eval()
        # epoch_dev_loss = 0.0
        # for dev_step, (batch_x, batch_y) in enumerate(dev_loader):
        #     b_x = batch_x.to(device)
        #     b_y = batch_y.to(device)
        #     if rdm_matching and not isinstance(net, torch.nn.Sequential):
        #         prediction = net(b_x, i=2)  # input x and predict based on x
        #     else:
        #         prediction = net(b_x)  # input x and predict based on x
        #
        #     if rdm_matching:
        #         rdm = compute_cov_matrix(prediction, prediction)
        #         kernel_matrix = torch.tensor(
        #             fitted_kernel(b_x.cpu().numpy(), b_x.cpu().numpy()),
        #             dtype=torch.float,
        #         ).to(device)
        #         #             loss = loss_func(torch.exp(rdm.reshape((-1))), torch.exp(kernel_matrix.reshape((-1))))    # must be (1. nn output, 2. target)
        #         loss = loss_func(
        #             rdm.reshape((-1)), kernel_matrix.reshape((-1))
        #         )  # must be (1. nn output, 2. target)
        #     else:
        #         loss = loss_func(
        #             prediction.reshape((-1)), b_y
        #         )  # must be (1. nn output, 2. target)
        #
        #     epoch_dev_loss += loss.item()
        #     t.set_postfix(
        #         epoch_loss=epoch_loss / (step + 1),
        #         epoch_dev_loss=epoch_dev_loss / (dev_step + 1),
        #     )
        if scheduler:
            scheduler.step()
        if epoch_loss < min_loss:
            torch.save(net.state_dict(), "best_model.pth")


def evaluate_net(net, X_train, Y_train, X_plot, Y_plot, device, save=""):
    net.load_state_dict(torch.load("best_model.pth"))
    net.eval()
    plt.plot(X_plot, Y_plot, color="orange", lw=2, label="True")
    plt.plot(X_train, Y_train, color="red", label="Traning data")
    # plt.plot(X_plot, np.sin(X_plot), color="orange", lw=2, label="True")
    prediction = net(
        torch.tensor(X_plot, dtype=torch.float).to(device)
    )  # input x and predict based on x
    plot_nn(prediction.detach().cpu().numpy(), X_plot, save=save)
    fig = plt.gcf()


#     fig.savefig("sin_net_match_periodic_retrained.png", dpi=200)
