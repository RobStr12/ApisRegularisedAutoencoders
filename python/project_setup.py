import os

os.system("python -m pip install --upgrade pip")
os.system("pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html")
os.system("pip install openpyxl")
os.system("pip install matplotlib")
os.system("pip install requests")
os.system("pip install tqdm")

import requests

papers = ["SP", "TH"]
folders = ["data", "model", "plots"]

paths = ["./data"]
paths.extend([f"./data/{paper}" for paper in papers])
paths.extend([f"./data/{paper}/{folder}" for paper in papers for folder in folders])
paths.extend([f"./data/{paper}/{folder}/{sub}" for paper in papers for folder in ["model", "plots"] for sub in ["simple", "l1", "denoise"]])

for path in paths:
    if not os.path.exists(path):
        os.mkdir(path)
        print(f"{path} created")
    else:
        print(f"{path} already exists.")

files = [("https://raw.githubusercontent.com/RobStr12/ApisRegularisedAutoencoders/main/data/raw/count_matrix_SP_raw.csv", "./data/SP/data/count_matrix_SP_raw.csv"),
         ("https://raw.githubusercontent.com/RobStr12/ApisRegularisedAutoencoders/main/data/raw/S3A%20DESeq.CK_vs_T.csv", "./data/TH/data/CK_vs_T.csv"),
         ("https://raw.githubusercontent.com/RobStr12/ApisRegularisedAutoencoders/main/data/raw/S3B%20DESeq.CK_vs_RH.csv", "./data/TH/data/CK_vs_RH.csv"),
         ("https://raw.githubusercontent.com/RobStr12/ApisRegularisedAutoencoders/main/data/raw/S3C%20DESeq.CK_vs_TH.csv", "./data/TH/data/CK_vs_TH.csv")]

for url, file_path in files:
    r = requests.get(url)
    handle = open(file_path, "w")
    handle.write("\n".join(r.text.rstrip().split()))
