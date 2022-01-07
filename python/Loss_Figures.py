import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plot_paths = [f"./data/TH/plots/denoise/model_{dl * 2 + 1}L_denoise_lr{lr}.png" for lr in ["0.0001", "5e-05", "1e-05", "5e-06", "1e-06"] for dl in range(3)]

for i, plot in enumerate(plot_paths):
    plt.subplot(5, 3, i+1)
    plt.imshow(mpimg.imread(plot))
    plt.axis("off")
    if i == 0:
        plt.title("1 hidden layer")
    elif i == 1:
        plt.title("3 hidden layers")
    elif i == 2:
        plt.title("5 hidden layers")
plt.show()
