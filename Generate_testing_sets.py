import pandas as pd
import os
import os

import pandas as pd

if __name__ == "__main__":
    path = "./resources/"
    os.chdir(path)
    newfolder = "ExperimentalData"
    if not os.path.exists(newfolder):
        os.makedirs(newfolder)

    for file in os.listdir():

        if file.endswith(".tsv"):
            temp = pd.read_csv(file, delimiter="\t", index_col="id").sort_values(
                "id")
            temp2 = temp.sample(5000, random_state=123)
            newfilename = os.path.join(newfolder, file)
            temp2.to_csv(f"{newfilename}.tsv", sep="\t")
