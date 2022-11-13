import csv
import collections
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


if __name__ == "__main__":

    sought_line = None
    input_query = "kv6loraw3A6MdnXd"

    with open("./resources/id_lyrics_tf-idf_mmsr.tsv") as file:
        tsv_file = csv.reader(file, delimiter="\t")
        # skip first line as is is the string headers
        next(tsv_file)
        for line in tsv_file:
            if line[0] == input_query:
                # Finding full line of our searched query
                sought_line = line
                break

    if sought_line is None:
        raise NameError("The searched input is not in the database")

    cosines = dict()

    with open("./resources/id_lyrics_tf-idf_mmsr.tsv") as file:
        tsv_file = csv.reader(file, delimiter="\t")
        # skip first line as is is the string headers
        next(tsv_file)
        for line in tsv_file:
            cosines[line[0]] = cosine_similarity(np.array([sought_line[1:]]), np.array([line[1:]]))[0, 0]

    print(cosines)

    # order cosines on value
    sorted_cosines = collections.OrderedDict(sorted(cosines.items(), key=lambda kv: kv[1], reverse=True))
    print("\n-----------------------------------The stuff we want ----------------------------------------------\n")
    print(sorted_cosines)
    print(type(sorted_cosines))
