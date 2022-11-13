import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import csv

if __name__ == "__main__":

    sought_line = 0
    input_query = "kv6loraw3A6MdnXd"

    with open("./resources/id_lyrics_tf-idf_mmsr.tsv") as file:
        tsv_file = csv.reader(file, delimiter="\t")
        # skip first line as is is the string headers
        next(tsv_file)
        for line in tsv_file:
            if line[0] == input_query:
                print(line)
                sought_line = line
                break

    print("Lol other stuff --------------------------------------------------\n")

    with open("./resources/id_lyrics_tf-idf_mmsr.tsv") as file:
        tsv_file = csv.reader(file, delimiter="\t")
        # skip first line as is is the string headers
        next(tsv_file)
        for line in tsv_file:
            if cosine_similarity(np.array([sought_line[1:]]), np.array([line[1:]])) == np.array([1.0]):
                print(line)








    # region dead code
    #for line in tsv_file:
    #    print(cosine_similarity(np.array(sought_line[1:]), np.transpose(np.array(line[1:]))))

    # if cosine_similarity(np.array([sought_line[1:]]), np.array([line[1:]])) == np.array([[1.0]]):
    #    print(line)


        # kv6loraw3A6MdnXd never gonna give you up

        # metadata = pd.read_csv("C:\Uni\2022W\MMSR\id_information_mmsr.tsv", sep="\t")
        # tf_idf = pd.read_csv("C:\Uni\2022W\MMSR\id_lyrics_tf-idf_mmsr.tsv", sep="\t")

        # print(metadata.where(metadata['id'] == 'kv6loraw3A6MdnXd'))
        # print(metadata[where 'id' == 'kv6loraw3A6MdnXd']))

        # print(metadata)
        # print(cosine_similarity(metadata, tsv_file))
    #endregion