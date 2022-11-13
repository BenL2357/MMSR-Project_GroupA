import collections
import csv
from ast import literal_eval

import numpy as np

if __name__ == "__main__":
    cosines = dict()

    cosines["a"] = 1
    cosines["b"] = 2
    cosines["c"] = 3
    cosines["d"] = 4
    cosines["e"] = 5
    cosines["f"] = 6
    cosines["g"] = 7
    cosines["h"] = 8
    cosines["i"] = 9
    cosines["j"] = 10

    genres = []
    input_genres = []

    with open("./resources/id_genres_mmsr.tsv") as file:
        tsv_file = csv.reader(file, delimiter="\t")
        next(tsv_file)

        for i in range(3, 4):

            line = next(tsv_file)
            line = next(tsv_file)
            line = next(tsv_file)
            line = next(tsv_file)
            print(line)
            input_genres.append(set(literal_eval(line[1])))

    print(input_genres)