import os
import random

import pandas as pd

if __name__ == "__main__":
    path = "./resources/"
    os.chdir(path)
    newfolder = "ExperimentalData"
    if not os.path.exists(newfolder):
        os.makedirs(newfolder)

    n = 68000
    s = 5000

    random.seed(123)
    vals = sorted(random.sample(range(1, n + 1), s))

    for file in os.listdir():

        if file.endswith(".tsv"):
            temp = pd.read_csv(file, delimiter="\t", index_col="id").sort_values(
                "id")
            temp2 = temp.iloc[vals, :]
            newfilename = os.path.join(newfolder, file)
            temp2.to_csv(newfilename, sep="\t")
