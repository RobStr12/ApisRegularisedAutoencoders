handle = open("./data/TH/data/count_matrix_TH.csv", "r")
text = handle.read()
handle.close()

lines = text.rstrip().split()[1:11]

table = []
for line in lines:
    row = list(map(float, line.split(",")))
    row = list(map(round, row))
    row = list(map(str, row))
    table.append(" & ".join(row))

table = [row + r" \\" for row in table]
table = "\n".join(table)
print(table)

handle = open("./paper/tableTH.txt", "w")
handle.write(table)
handle.close()
