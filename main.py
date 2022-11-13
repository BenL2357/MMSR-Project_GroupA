import collections
import csv

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

if __name__ == "__main__":

    sought_line_cosine = None
    sought_line_euclidean = None
    input_query = "kv6loraw3A6MdnXd"


    with open("./resources/id_lyrics_tf-idf_mmsr.tsv") as file:
        tsv_file = csv.reader(file, delimiter="\t")
        # skip first line as it is the string headers
        next(tsv_file)
        for line in tsv_file:
            if line[0] == input_query:
                # Finding full line of our searched query
                sought_line_cosine = line
                break

    with open("./resources/id_bert_mmsr.tsv") as file:
        tsv_file = csv.reader(file, delimiter="\t")
        # skip first line as it is the string headers
        next(tsv_file)
        for line in tsv_file:
            if line[0] == input_query:
                # Finding full line of our searched query
                sought_line_bert = line
                break

    if sought_line_cosine is None or sought_line_bert is None:
        raise NameError("The searched input is not in the database")

    # using tfidf
    cosines = dict()
    # using bert
    euclideans = dict()

    with open("./resources/id_lyrics_tf-idf_mmsr.tsv") as file:
        tsv_file = csv.reader(file, delimiter="\t")
        # skip first line as it is the string headers
        next(tsv_file)
        for line in tsv_file:
            cosines[line[0]] = cosine_similarity(np.array([sought_line_cosine[1:]]), np.array([line[1:]]))[0, 0]

    with open("./resources/id_bert_mmsr.tsv") as file:
        tsv_file = csv.reader(file, delimiter="\t")
        # skip first line as it is the string headers
        next(tsv_file)
        for line in tsv_file:
            euclideans[line[0]] = (1 / (1 + euclidean_distances(np.array([sought_line_bert[1:]]), np.array([line[1:]]), squared=True)))[0, 0]

    # order dictionaries
    sorted_cosines = collections.OrderedDict(sorted(cosines.items(), key=lambda kv: kv[1], reverse=True))
    sorted_euclideans = collections.OrderedDict(sorted(euclideans.items(), key=lambda kv: kv[1], reverse=True))


