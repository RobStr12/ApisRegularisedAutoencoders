from tqdm import tqdm

print("Cleaning up and rearranging raw count matrix of SP")

handle = open("./data/SP/data/count_matrix_SP_raw.csv", "r")
text = handle.read()
handle.close()

lines = text.rstrip().split()
data = {}

for line in lines[1:]:
    d = line.split(",")
    id = int(d[0][1:-1])
    data[id] = d[1:]

keys = sorted(list(data.keys()))

handle = open("./data/SP/data/count_matrix_SP.csv", "w")
handle.write(lines[0] + "\n")

for key in keys:
    handle.write(f"{str(key)}, {','.join(data[key])}\n")
handle.close()

print("Constructing the count matrix for TH")

files = [f"./data/TH/data/CK_vs_{tr}.csv" for tr in ["T", "RH", "TH"]]

cmat = {}
treatments = []

treatments
for j, file in enumerate(files):

    handle = open(file, "r")
    text = handle.read()
    handle.close()

    lines = text.rstrip().split()

    for i, line in tqdm(enumerate(lines), total=len(lines)):
        data = line.split(",")
        id = data[0]
        tr1 = data[2]
        tr2 = data[3]
        if i == 0:
            treats = [tr1 + str(j + 1), tr2]
        elif data[0] in cmat.keys():
            cmat[id][treats[0]] = tr1
            cmat[id][treats[1]] = tr2
        else:
            cmat[id] = {treats[0]: tr1, treats[1]: tr2}

    treatments.extend(treats)

ids = list(map(str, sorted(list(map(int, cmat.keys())))))

header = ["id"]
header.extend(treatments)
text = [",".join(header)]

for id in tqdm(ids, total = len(ids)):
    raw = cmat[id]
    data = [id]
    for tr in treatments:
        if tr in raw.keys():
            data.append(str(raw[tr]))
        else:
            data.append("0")

    text.append(",".join(data))

handle = open("./data/TH/data/count_matrix_TH.csv", "w")
handle.write("\n".join(text))
handle.close()