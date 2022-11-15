import csv
from ast import literal_eval

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

if __name__ == "__main__":

    def sim_query(input_query: str, nvals: int = 100):
        sought_line_cosine = None
        sought_line_euclidean = None

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
                    sought_line_euclidean = line
                    break

        if sought_line_cosine is None or sought_line_euclidean is None:
            raise NameError("The searched input is not in the database")

        similarities = dict()

        with open("./resources/id_lyrics_tf-idf_mmsr.tsv") as file:
            tsv_file = csv.reader(file, delimiter="\t")
            # skip first line as it is the string headers
            next(tsv_file)
            for line in tsv_file:
                similarities[line[0]] = cosine_similarity(np.array([sought_line_cosine[1:]]), np.array([line[1:]]))[
                    0, 0]

        with open("./resources/id_bert_mmsr.tsv") as file:
            tsv_file = csv.reader(file, delimiter="\t")
            # skip first line as it is the string headers
            next(tsv_file)
            for line in tsv_file:
                similarities[line[0]] = similarities[line[0]] * 0.5 + \
                                        (1 / (1 + euclidean_distances(np.array([sought_line_euclidean[1:]]),
                                                                      np.array([line[1:]]),
                                                                      squared=True)))[0, 0] * 0.5

        # removing the search query fom the results
        similarities.pop(input_query, None)
        # ordering them by their similarity
        similarities = sorted(similarities.items(), key=lambda kv: kv[1],
                              reverse=True)  # dict(sorted(similarities.items(), key=lambda kv: kv[1], reverse=True))

        return [x[0] for x in similarities[:nvals]]


    def sim_query_with_relevance(input_query: str, nvals: int = 100):
        sought_line_cosine = None
        sought_line_euclidean = None

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
                    sought_line_euclidean = line
                    break

        if sought_line_cosine is None or sought_line_euclidean is None:
            raise NameError("The searched input is not in the database")

        similarities = dict()

        with open("./resources/id_lyrics_tf-idf_mmsr.tsv") as file:
            tsv_file = csv.reader(file, delimiter="\t")
            # skip first line as it is the string headers
            next(tsv_file)
            for line in tsv_file:
                similarities[line[0]] = cosine_similarity(np.array([sought_line_cosine[1:]]), np.array([line[1:]]))[
                    0, 0]

        with open("./resources/id_bert_mmsr.tsv") as file:
            tsv_file = csv.reader(file, delimiter="\t")
            # skip first line as it is the string headers
            next(tsv_file)
            for line in tsv_file:
                similarities[line[0]] = similarities[line[0]] * 0.5 + \
                                        (1 / (1 + euclidean_distances(np.array([sought_line_euclidean[1:]]),
                                                                      np.array([line[1:]]),
                                                                      squared=True)))[0, 0] * 0.5

        # removing the search query fom the results
        similarities.pop(input_query, None)
        # ordering them by their similarity
        similarities = sorted(similarities.items(), key=lambda kv: kv[1],
                              reverse=True)  # dict(sorted(similarities.items(), key=lambda kv: kv[1], reverse=True))

        return similarities[:nvals]


    def precision(input_query: str, results: [str]):

        genres = []
        input_genres = []

        with open("./resources/id_genres_mmsr.tsv") as file:
            tsv_file = csv.reader(file, delimiter="\t")
            # skip first line as it is the string headers
            next(tsv_file)
            for line in tsv_file:
                if line[0] == input_query:
                    input_genres = set(literal_eval(line[1]))
                if line[0] in results:
                    genres.append(set(literal_eval(line[1])))

        hits = 0
        for val in genres:
            if len(input_genres & val) != 0:
                hits += 1

        return hits / len(results)


    #    def genre_as_set(line: str):
    #        return set(literal_eval(line))

    def MRR(input_query: str, results: [str]):

        genres = []
        input_genres = []

        with open("./resources/id_genres_mmsr.tsv") as file:
            tsv_file = csv.reader(file, delimiter="\t")
            # skip first line as it is the string headers
            next(tsv_file)
            for line in tsv_file:
                if line[0] == input_query:
                    input_genres = set(literal_eval(line[1]))
                if line[0] in results:
                    genres.append(set(literal_eval(line[1])))

        tries = 1
        for val in genres:
            temp = input_genres & val
            if len(temp) == 0:
                tries += 1
            else:
                break

        return 1 / tries


    def nDCG(results: [(str, float)], p: int):
        relevance = [x[1] for x in results]
        relevance_unordered = [x[1] for x in sorted(results, key=lambda x: x[0])]

        DCG = 0

        for i in range(0, p + 1):
            DCG += (pow(2, relevance_unordered[i]) - 1) / np.log2(i + 2)

        IDCG = 0
        index2 = 0
        for val in relevance:
            IDCG += (pow(2, val) - 1) / (np.log2(index2 + 2))
            index2 += 1

        return DCG / IDCG


    def performance_metrics():
        query_song_strings = ["D6xz7HXyidDcseg4", "3xza64DGjULmd7ws", "nl0KhzV5YRUzlbQy", "zCZAncx2kD58UGGA",
                              "3yOJLIpOiYvqCKAm",
                              "4JEHIYg24AFHVEe4", "tXhoFYt3gUf38kHN", "zZl3n47f7kW5ygrv", "J4iv1rHSF2ky0K4n",
                              "2YKPm6gHu6CRWeyx"]
        precision_sum = 0
        mrr_sum = 0
        ndcg_sum_10 = 0
        ndcg_sum_100 = 0
        for query_string in query_song_strings:
            result = sim_query_with_relevance(query_string)
            result_similarity_score = [x[1] for x in result]
            precision_sum += precision(query_string, result_similarity_score)
            mrr_sum += MRR(query_string, result_similarity_score)
            ndcg_sum_10 += nDCG(result, 10)
            ndcg_sum_100 += nDCG(result, 100)

        return precision_sum / len(query_song_strings), mrr_sum / len(query_song_strings), ndcg_sum_10 / len(query_song_strings), ndcg_sum_100 / len(query_song_strings)


    print(nDCG(sim_query_with_relevance("kv6loraw3A6MdnXd"), 10))
