import argparse
import torch
from ApisDataset import ApisDataset
from ApisNeuralNet import *
from torch import optim
import time
import matplotlib.pyplot as plt
import numpy as np

# construct the parser

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--paper", type=str, default="TH", help="Paper for which a neural net should be constructed")
ap.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs for the training of the net.")
ap.add_argument("-rp", "--reg_param", type=float, default=0.001,
                help="Regression Parameter for the different autoencoders:\n"
                     "-simple autoencoder: not applied.\n"
                     "-sparse autoencoder with L1 regularization: lambda.\n"
                     "-sparse autoencoder with KL-divergence: beta.\n"
                     "-denoising autoencoder: noise factor.")
ap.add_argument("-r", "--rho", type=float, default=0.05, help="Rho value for the KL-divergence")
ap.add_argument("-d", "--device", type=str, default="cpu", help="Device for calculations if available. Default is cpu")
args = vars(ap.parse_args())

paper = args["paper"]
epochs = args["epochs"]
reg_param = args["reg_param"]
RHO = args["rho"]
device = "cuda" if args["device"] == "cuda" and torch.cuda.is_available() else "cpu"

# loading in dataset
apis_data = ApisDataset(paper=paper, device=device)

# setting up models and base loss
models = [ApisNeuralNet1, ApisNeuralNet3, ApisNeuralNet5, ApisNeuralNet31, ApisNeuralNet51, ApisNeuralNet53]
criterion = nn.MSELoss()


# setting up loss functions
def sparse_l1_loss(model_children, profile):
    loss = 0
    values = profile
    for i in range(len(model_children)):
        values = torch.relu((model_children[i](values)))
        loss += torch.mean(torch.abs(values))
    return loss


def fit(model, dataset, aetype, optimizer):
    print(f"Training")
    model.train()
    model_children = list(model.children())

    running_loss = 0.0
    counter = 0

    for data in dataset:
        counter += 1
        if aetype == "denoise":
            profile = data + reg_param * torch.randn(data.shape).to(device)
        else:
            profile = data

        profile = profile.to(device)
        optimizer.zero_grad()
        outputs = model(profile)
        mse_loss = criterion(outputs, profile)

        if aetype == "l1":
            l1_loss = sparse_l1_loss(model_children, profile)
            loss = mse_loss + reg_param * l1_loss
        else:
            loss = mse_loss

        loss = loss.cpu()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / counter
    print(f"Epoch loss: {epoch_loss}")

    return epoch_loss


def validate(model, dataset, aetype):
    print("Validating")
    model.eval()

    running_loss = 0.0
    counter = 0

    with torch.no_grad():
        for data in dataset:
            counter += 1

            if aetype == "denoise":
                profile = data + reg_param * torch.randn(data.shape).to(device)
            else:
                profile = data

            profile = profile.to(device)
            outputs = model(profile)
            loss = criterion(outputs, profile)
            running_loss += loss.cpu()

        epoch_loss = running_loss / counter
        print(f"Epoch loss: {epoch_loss}")

        return epoch_loss


aetypes = ["simple", "l1", "denoise"]
learning_rates = [0.0001, 0.00005, 0.00001, 0.000005, 0.000001]
plots = {}

for learning_rate in learning_rates:
    for aetype in aetypes:
        print(aetype)
        for i, mdl in enumerate(models):
            model = mdl(apis_data.n_genes, 2).to(device)
            print(model)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            train_loss = []
            val_loss = []
            start = time.time()

            for epoch in range(epochs):
                print(f"Epoch {epoch + 1}/{epochs}")
                train_epoch_loss = fit(model, apis_data, aetype, optimizer)
                val_epoch_loss = validate(model, apis_data, aetype)

                train_loss.append(train_epoch_loss)
                val_loss.append(val_epoch_loss)

            end = time.time()

            print(f"Time: {(end - start) // 60}:{(end - start) % 60:.2f}")

            torch.save(model.state_dict(), f"./data/{paper}/model/{aetype}/model_{i + 1}_{aetype}_lr{learning_rate}.pth")

            plt.figure(figsize=(10, 7))
            plt.plot(train_loss, color="orange", label="training loss")
            plt.plot(val_loss, color="yellow", label="validation loss")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend()
            plt.savefig(f"./data/{paper}/plots/{aetype}/model_{i + 1}_{aetype}_lr{learning_rate}.png")

            dl = str(i + 1)
            if aetype in plots.keys():
                print(dl)
                if dl in plots[aetype].keys():
                    plots[aetype][dl][learning_rate] = (train_loss, val_loss)
                else:
                    plots[aetype][dl] = {learning_rate: (train_loss, val_loss)}
            else:
                plots[aetype] = {}
                plots[aetype][dl] = {}
                plots[aetype][dl][learning_rate] = (train_loss, val_loss)


for aetype in aetypes:
    plt.figure()
    i = 0
    for learning_rate in learning_rates:
        for dl in [str(i + 1) for i in range(6)]:
            i += 1
            plt.subplot(5,5,i)
            dt = plots[aetype][dl][learning_rate]
            plt.plot(dt[0], color ="orange", label="training loss")
            plt.plot(dt[1], color="yellow", label="validation loss")
            plt.yticks([])
            plt.yticks([])

    plt.savefig(f"./data/{paper}/plots/{aetype}/summary.png")
