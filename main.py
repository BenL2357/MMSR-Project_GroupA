#import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#kv6loraw3A6MdnXd never gonna give you up

#metadata = pd.read_csv("C:\Uni\2022W\MMSR\id_information_mmsr.tsv", sep="\t")
#tf_idf = pd.read_csv("C:\Uni\2022W\MMSR\id_lyrics_tf-idf_mmsr.tsv", sep="\t")

#print(metadata.where(metadata['id'] == 'kv6loraw3A6MdnXd'))
#print(metadata[where 'id' == 'kv6loraw3A6MdnXd']))

#print(metadata)
#print(cosine_similarity(metadata, tsv_file))

import csv


if __name__ == "__main__":
    sought_line = 0

    with open("./resources/id_lyrics_tf-idf_mmsr.tsv") as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line in tsv_file:
            if(line[0] == "kv6loraw3A6MdnXd"):
                print(line)
                sought_line = line
                break

    #for line in tsv_file:
    #    print(cosine_similarity(np.array(sought_line[1:]), np.transpose(np.array(line[1:]))))