import torch
from ApisDataset import ApisDataset
from ApisNeuralNet import *
import matplotlib.pyplot as plt
import time

start = time.time()

papers = ["TH", "SP"]
aet = ["simple", "l1", "denoise"]
layers = [str(i + 1) for i in range(6)]
W = [0, 1, 2, 3, 12, 13, 14, 15]
F = [4, 5, 6, 7, 16, 17, 18, 19]

for paper in papers:
    plt.figure()
    i = 0
    apis_data = ApisDataset(paper)
    apis_models = [ApisNeuralNet1(apis_data.n_genes, 2),
                   ApisNeuralNet3(apis_data.n_genes, 2),
                   ApisNeuralNet5(apis_data.n_genes, 2),
                   ApisNeuralNet31(apis_data.n_genes, 2),
                   ApisNeuralNet51(apis_data.n_genes, 2),
                   ApisNeuralNet53(apis_data.n_genes, 2)]
    for j, dl in enumerate(layers):
        for ae in aet:
            i += 1
            model = apis_models[j]
            model.load_state_dict(torch.load(f"./data/{paper}/model/{ae}/model_{dl}_{ae}_lr5e-05.pth"))

            outputs = model.encode(apis_data.tensor)
            print(f"Paper: {paper}; Number of hidden layers: {dl}; Autoencoder type: {ae}")

            plt.subplot(6, 3, i)
            for k, point in enumerate(outputs.detach().numpy()):
                x, y = point

                if k < 12:
                    shape = "o"
                else:
                    shape = "v"

                if paper == "TH":
                    if k % 2 == 0:
                        color = "#EBA937"
                    else:
                        color = "#EB4F37"
                else:
                    if k in W:
                        color = "#EBA937"
                    elif k in F:
                        color = "#D3EB37"
                    else:
                        color = "#EB4F37"

                print(f"Colour: {color}; Shape: {shape}")

                plt.plot(x, y, shape, color=color)

            plt.xticks([])
            plt.yticks([])

    plt.savefig(f"./data/{paper}/results.png")

end = time.time()
run = end - start

print(f"{int(run // 60)}:{run % 60:.2f}")
