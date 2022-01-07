from torch import nn


class ApisNeuralNet1(nn.Module):

    def __init__(self, input_size, latent_space):
        super(ApisNeuralNet1, self).__init__()

        self.relu = nn.ReLU()

        self.enc1 = nn.Linear(input_size, input_size // 2)
        self.enc2 = nn.Linear(input_size // 2, latent_space)

        self.dec1 = nn.Linear(latent_space, input_size // 2)
        self.dec2 = nn.Linear(input_size // 2, input_size)

    def encode(self, x):
        x = self.enc1(x)
        x = self.relu(x)
        x = self.enc2(x)
        x = self.relu(x)
        return x

    def decode(self, x):
        x = self.dec1(x)
        x = self.relu(x)
        x = self.dec2(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


class ApisNeuralNet3(nn.Module):

    def __init__(self, input_size, latent_space):
        super(ApisNeuralNet3, self).__init__()

        self.relu = nn.ReLU()

        self.enc1 = nn.Linear(input_size, input_size // 2)
        self.enc2 = nn.Linear(input_size // 2, input_size // 4)
        self.enc3 = nn.Linear(input_size // 4, input_size // 8)
        self.enc4 = nn.Linear(input_size // 8, latent_space)

        self.dec1 = nn.Linear(latent_space, input_size // 8)
        self.dec2 = nn.Linear(input_size // 8, input_size // 4)
        self.dec3 = nn.Linear(input_size // 4, input_size // 2)
        self.dec4 = nn.Linear(input_size // 2, input_size)

    def encode(self, x):
        x = self.enc1(x)
        x = self.relu(x)
        x = self.enc2(x)
        x = self.relu(x)
        x = self.enc3(x)
        x = self.relu(x)
        x = self.enc4(x)
        x = self.relu(x)
        return x

    def decode(self, x):
        x = self.dec1(x)
        x = self.relu(x)
        x = self.dec2(x)
        x = self.relu(x)
        x = self.dec3(x)
        x = self.relu(x)
        x = self.dec4(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


class ApisNeuralNet5(nn.Module):

    def __init__(self, input_size, latent_space):
        super(ApisNeuralNet5, self).__init__()

        self.relu = nn.ReLU()

        self.enc1 = nn.Linear(input_size, input_size // 2)
        self.enc2 = nn.Linear(input_size // 2, input_size // 4)
        self.enc3 = nn.Linear(input_size // 4, input_size // 8)
        self.enc4 = nn.Linear(input_size // 8, input_size // 16)
        self.enc5 = nn.Linear(input_size // 16, input_size // 32)
        self.enc6 = nn.Linear(input_size // 32, latent_space)

        self.dec1 = nn.Linear(latent_space, input_size // 32)
        self.dec2 = nn.Linear(input_size // 32, input_size // 16)
        self.dec3 = nn.Linear(input_size // 16, input_size // 8)
        self.dec4 = nn.Linear(input_size // 8, input_size // 4)
        self.dec5 = nn.Linear(input_size // 4, input_size // 2)
        self.dec6 = nn.Linear(input_size // 2, input_size)

    def encode(self, x):
        x = self.enc1(x)
        x = self.relu(x)
        x = self.enc2(x)
        x = self.relu(x)
        x = self.enc3(x)
        x = self.relu(x)
        x = self.enc4(x)
        x = self.relu(x)
        x = self.enc5(x)
        x = self.relu(x)
        x = self.enc6(x)
        x = self.relu(x)
        return x

    def decode(self, x):
        x = self.dec1(x)
        x = self.relu(x)
        x = self.dec2(x)
        x = self.relu(x)
        x = self.dec3(x)
        x = self.relu(x)
        x = self.dec4(x)
        x = self.relu(x)
        x = self.dec5(x)
        x = self.relu(x)
        x = self.dec6(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


class ApisNeuralNet31(nn.Module):

    def __init__(self, input_size, latent_space):
        super(ApisNeuralNet31, self).__init__()

        self.relu = nn.ReLU()

        self.enc1 = nn.Linear(input_size, input_size // 2)
        self.enc2 = nn.Linear(input_size // 2, input_size // 4)
        self.enc3 = nn.Linear(input_size // 4, input_size // 8)
        self.enc4 = nn.Linear(input_size // 8, latent_space)

        self.dec1 = nn.Linear(latent_space, input_size // 2)
        self.dec2 = nn.Linear(input_size // 2, input_size)

    def encode(self, x):
        x = self.enc1(x)
        x = self.relu(x)
        x = self.enc2(x)
        x = self.relu(x)
        x = self.enc3(x)
        x = self.relu(x)
        x = self.enc4(x)
        x = self.relu(x)
        return x

    def decode(self, x):
        x = self.dec1(x)
        x = self.relu(x)
        x = self.dec2(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


class ApisNeuralNet51(nn.Module):

    def __init__(self, input_size, latent_space):
        super(ApisNeuralNet51, self).__init__()

        self.relu = nn.ReLU()

        self.enc1 = nn.Linear(input_size, input_size // 2)
        self.enc2 = nn.Linear(input_size // 2, input_size // 4)
        self.enc3 = nn.Linear(input_size // 4, input_size // 8)
        self.enc4 = nn.Linear(input_size // 8, input_size // 16)
        self.enc5 = nn.Linear(input_size // 16, input_size // 32)
        self.enc6 = nn.Linear(input_size // 32, latent_space)

        self.dec1 = nn.Linear(latent_space, input_size // 2)
        self.dec2 = nn.Linear(input_size // 2, input_size)

    def encode(self, x):
        x = self.enc1(x)
        x = self.relu(x)
        x = self.enc2(x)
        x = self.relu(x)
        x = self.enc3(x)
        x = self.relu(x)
        x = self.enc4(x)
        x = self.relu(x)
        x = self.enc5(x)
        x = self.relu(x)
        x = self.enc6(x)
        x = self.relu(x)
        return x

    def decode(self, x):
        x = self.dec1(x)
        x = self.relu(x)
        x = self.dec2(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


class ApisNeuralNet53(nn.Module):

    def __init__(self, input_size, latent_space):
        super(ApisNeuralNet53, self).__init__()

        self.relu = nn.ReLU()

        self.enc1 = nn.Linear(input_size, input_size // 2)
        self.enc2 = nn.Linear(input_size // 2, input_size // 4)
        self.enc3 = nn.Linear(input_size // 4, input_size // 8)
        self.enc4 = nn.Linear(input_size // 8, input_size // 16)
        self.enc5 = nn.Linear(input_size // 16, input_size // 32)
        self.enc6 = nn.Linear(input_size // 32, latent_space)

        self.dec1 = nn.Linear(latent_space, input_size // 8)
        self.dec2 = nn.Linear(input_size // 8, input_size // 4)
        self.dec3 = nn.Linear(input_size // 4, input_size // 2)
        self.dec4 = nn.Linear(input_size // 2, input_size)

    def encode(self, x):
        x = self.enc1(x)
        x = self.relu(x)
        x = self.enc2(x)
        x = self.relu(x)
        x = self.enc3(x)
        x = self.relu(x)
        x = self.enc4(x)
        x = self.relu(x)
        x = self.enc5(x)
        x = self.relu(x)
        x = self.enc6(x)
        x = self.relu(x)
        return x

    def decode(self, x):
        x = self.dec1(x)
        x = self.relu(x)
        x = self.dec2(x)
        x = self.relu(x)
        x = self.dec3(x)
        x = self.relu(x)
        x = self.dec4(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
