import ast
import csv
from ast import literal_eval

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

if __name__ == "__main__":

    # region Memory Inefficient but fast

    def sim_query_m(input_query: str, nvals: int = 100):

        tf_idf_df = pd.read_csv('./resources/id_lyrics_tf-idf_mmsr.tsv', delimiter="\t", index_col="id")
        bert_df = pd.read_csv('./resources/id_bert_mmsr.tsv', delimiter="\t", index_col="id")

        similarity = cosine_similarity(tf_idf_df.loc[[input_query]], tf_idf_df)[0] * 0.5 + \
                     (1 / (1 + euclidean_distances(bert_df.loc[[input_query]], bert_df, squared=True)[
                         0])) * 0.5

        res = pd.DataFrame(index=tf_idf_df.index.tolist())
        res.index.name = "id"
        res["similarity"] = similarity
        res.drop([input_query], axis=0, inplace=True)

        return res.nlargest(nvals, "similarity")


    def precision_m(input_query: str, results):
        genres = pd.read_csv('./resources/id_genres_mmsr.tsv', delimiter="\t", index_col="id")
        genres["genre"] = genres["genre"].apply(ast.literal_eval)

        input_genres = genres.loc[input_query]["genre"]

        hits = 0
        for index, row in results.iterrows():
            if len(np.intersect1d(genres.loc[index]["genre"], input_genres)) >= 1:
                hits += 1

        return hits / len(results)


    def mrr_m(input_query: str, results):

        genres = pd.read_csv('./resources/id_genres_mmsr.tsv', delimiter="\t", index_col="id")
        genres["genre"] = genres["genre"].apply(ast.literal_eval)

        input_genres = genres.loc[input_query]["genre"]

        tries = 1
        for index, row in results.iterrows():
            if len(np.intersect1d(genres.loc[index]["genre"], input_genres)) >= 1:
                tries += 1
            else:
                break

        return 1 / tries


    # endregion

    # region Memory Efficient and fast

    def sim_query_s(tfidf_df, bert_df, input_query: str, nvals: int = 100):

        similarity = cosine_similarity(tfidf_df.loc[[input_query]], tfidf_df)[0] * 0.5 + \
                     (1 / (1 + euclidean_distances(bert_df.loc[[input_query]], bert_df, squared=True)[
                         0])) * 0.5

        res = pd.DataFrame(index=tfidf_df.index.tolist())
        res.index.name = "id"
        res["similarity"] = similarity
        res.drop([input_query], axis=0, inplace=True)

        return res.nlargest(nvals, "similarity")


    def precision_s(genres, input_query: str, results):

        input_genres = genres.loc[input_query]["genre"]

        hits = 0
        for index, row in results.iterrows():
            if len(np.intersect1d(genres.loc[index]["genre"], input_genres)) >= 1:
                hits += 1

        return hits / len(results)


    def mrr_s(genres, input_query: str, results):

        input_genres = genres.loc[input_query]["genre"]

        tries = 1
        for index, row in results.iterrows():
            if len(np.intersect1d(genres.loc[index]["genre"], input_genres)) >= 1:
                tries += 1
            else:
                break

        return 1 / tries


    def nDCG_ms(results, genres, input_query: str, p: int):

        input_genres = genres.loc[input_query]["genre"]
        relevance_scores = np.zeros((p))

        iter_index = 0
        for index, row in results.iterrows():
            if iter_index >= p:
                break
            if len(np.intersect1d(genres.loc[index]["genre"], input_genres)) >= 1:
                relevance_scores[iter_index] = 1
            iter_index += 1




        DCG = relevance_scores[0]
        IDCG = 1

        for i in range(1, p):
            DCG += relevance_scores[i] / np.log2(i + 1)
            IDCG += 1 / np.log2(i + 1)

        return DCG / IDCG


    def performance_metrics_s():
        precision_sum = 0
        mrr_sum = 0
        ndcg_sum_10 = 0
        ndcg_sum_100 = 0

        tfidf_df = pd.read_csv('./resources/id_lyrics_tf-idf_mmsr.tsv', delimiter="\t", index_col="id")
        bert_df = pd.read_csv('./resources/id_bert_mmsr.tsv', delimiter="\t", index_col="id")
        genres = pd.read_csv('./resources/id_genres_mmsr.tsv', delimiter="\t", index_col="id")

        genres["genre"] = genres["genre"].apply(ast.literal_eval)

        all_songs = tfidf_df.index.tolist()

        all_songs = all_songs[0:100]

        for song in all_songs:
            results = sim_query_s(tfidf_df, bert_df, song)
            precision_sum += precision_s(genres, song, results)
            mrr_sum += mrr_s(genres, song, results)
            ndcg_sum_10 += nDCG_ms(results, genres, song, 10)
            ndcg_sum_100 += nDCG_ms(results, genres, song, 100)

        nsongs = len(all_songs)

        return precision_sum / nsongs, mrr_sum / nsongs, ndcg_sum_10 / nsongs, ndcg_sum_100 / nsongs


    # endregion

    # region Memory Efficient but slow
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


    def mrr(input_query: str, results: [str]):

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
        IDCG = 0

        for i in range(0, p):
            DCG += (pow(2, relevance_unordered[i]) - 1) / np.log2(i + 2)
            IDCG += (pow(2, relevance[i]) - 1) / (np.log2(i + 2))

        return DCG / IDCG


    def performance_metrics():
        query_song_strings = ["J4iv1rHSF2ky0K4n", "2YKPm6gHu6CRWeyx"]
        precision_sum = 0
        mrr_sum = 0
        ndcg_sum_10 = 0
        ndcg_sum_100 = 0
        for query_string in query_song_strings:
            result = sim_query_with_relevance(query_string)
            result_similarity_score = [x[1] for x in result]
            precision_sum += precision(query_string, result_similarity_score)
            mrr_sum += mrr(query_string, result_similarity_score)
            ndcg_sum_10 += nDCG(result, 10)
            ndcg_sum_100 += nDCG(result, 100)

        return precision_sum / len(query_song_strings), mrr_sum / len(query_song_strings), ndcg_sum_10 / len(
            query_song_strings), ndcg_sum_100 / len(query_song_strings)


    # endregion

    precision_mean, mrr_mean, ndcg10_mean, ndcg100_mean = performance_metrics_s()
    print(f"Precision: {precision_mean}\n")
    print(f"MRR: {mrr_mean}\n")
    print(f"NDCG10 {ndcg10_mean}\n")
    print(f"NDCG100 {ndcg100_mean}\n")
