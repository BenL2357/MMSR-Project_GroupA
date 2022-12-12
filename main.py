import ast
import collections
import csv
from ast import literal_eval

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import PCA

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

        res_sim = res[res["similarity"] > 0.55]
        if len(res_sim) < nvals:
            res_sim = res.nlargest(nvals, "similarity")
        else:
            res_sim = res_sim.sort_values(by=["similarity"], ascending=False)

        return res_sim


    def sim_query_s_video(video_feature_vector, input_query: str, nvals: int = 100):
        similarity = cosine_similarity(video_feature_vector.loc[[input_query]], video_feature_vector)[0]

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
                break
            else:
                tries += 1

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


    def get_genre_distribution(genres):
        genre_dict = dict()

        for song_genres in genres["genre"]:
            for genre in song_genres:
                if genre in genre_dict:
                    genre_dict[genre] += 1
                else:
                    genre_dict[genre] = 1

        sorted_genre_dict = {k: v for k, v in sorted(genre_dict.items(), key=lambda item: item[1], reverse=True)}
        return sorted_genre_dict


    def avg_count_sharing_one_genre(genres):
        count = 0
        for genre_1 in genres["genre"]:
            for genre_2 in genres["genre"]:
                if len(np.intersect1d(genre_2, genre_1)) >= 1:
                    count += 1
        return count / len(genres)


    def performance_metrics_s(tfidf_df, bert_df, genres):
        precision_sum = 0
        mrr_sum = 0
        ndcg_sum_10 = 0
        ndcg_sum_100 = 0

        precision_sum_bl = 0
        mrr_sum_bl = 0
        ndcg_sum_10_bl = 0
        ndcg_sum_100_bl = 0

        # added sample
        all_songs = tfidf_df.sample(n=100, random_state=22031307).index.tolist()

        for song in all_songs:
            results = sim_query_s(tfidf_df, bert_df, song)
            random_results = tfidf_df.sample(n=100)
            # query values
            precision_sum += precision_s(genres, song, results)
            mrr_sum += mrr_s(genres, song, results)
            ndcg_sum_10 += nDCG_ms(results, genres, song, 10)
            ndcg_sum_100 += nDCG_ms(results, genres, song, 100)
            # baseline values
            precision_sum_bl += precision_s(genres, song, random_results)
            mrr_sum_bl += mrr_s(genres, song, random_results)
            ndcg_sum_10_bl += nDCG_ms(random_results, genres, song, 10)
            ndcg_sum_100_bl += nDCG_ms(random_results, genres, song, 100)

        nsongs = len(all_songs)

        return precision_sum / nsongs, mrr_sum / nsongs, ndcg_sum_10 / nsongs, ndcg_sum_100 / nsongs, precision_sum_bl / nsongs, mrr_sum_bl / nsongs, ndcg_sum_10_bl / nsongs, ndcg_sum_100_bl / nsongs

    def performance_metrics_s_baseline(tfidf_df, genres):
        precision_sum_bl = 0
        mrr_sum_bl = 0
        ndcg_sum_10_bl = 0
        ndcg_sum_100_bl = 0

        # added sample
        all_songs = tfidf_df.sample(random_state=22031307).index.tolist()

        rand_int = 70313022
        for song in all_songs:
            random_results = tfidf_df.sample(n=100, random_state=rand_int)
            # baseline values
            precision_sum_bl += precision_s(genres, song, random_results)
            mrr_sum_bl += mrr_s(genres, song, random_results)
            ndcg_sum_10_bl += nDCG_ms(random_results, genres, song, 10)
            ndcg_sum_100_bl += nDCG_ms(random_results, genres, song, 100)
            rand_int += 1

        nsongs = len(all_songs)

        print(f"Baseline Metric: \n")
        print(f"Precision Baseline: {precision_sum_bl / nsongs}\n")
        print(f"MRR Baseline: {mrr_sum_bl / nsongs}\n")
        print(f"NDCG10 Baseline: {ndcg_sum_10_bl / nsongs}\n")
        print(f"NDCG100 Baseline: {ndcg_sum_100_bl / nsongs}\n")


    def performance_metrics_s_video(video_features, genres):
        precision_sum = 0
        mrr_sum = 0
        ndcg_sum_10 = 0
        ndcg_sum_100 = 0

        all_songs = video_features.index.tolist()
        all_songs = all_songs[0:100]

        for song in all_songs:
            results = sim_query_s_video(video_features, song)
            precision_sum += precision_s(genres, song, results)
            mrr_sum += mrr_s(genres, song, results)
            ndcg_sum_10 += nDCG_ms(results, genres, song, 10)
            ndcg_sum_100 += nDCG_ms(results, genres, song, 100)

        print(f"{precision_sum / len(all_songs)}\n")
        print(f"{mrr_sum / len(all_songs)}\n")
        print(f"{ndcg_sum_10 / len(all_songs)}\n")
        print(f"{ndcg_sum_100 / len(all_songs)}\n")


    def initialize():
        tfidf_df = pd.read_csv('./resources/id_lyrics_tf-idf_mmsr.tsv', delimiter="\t", index_col="id")
        bert_df = pd.read_csv('./resources/id_lyrics_bert_mmsr.tsv', delimiter="\t", index_col="id")
        genres = pd.read_csv('./resources/id_genres_mmsr.tsv', delimiter="\t", index_col="id")

        video_features_resnet_output = pd.read_csv('./resources/id_resnet_mmsr.tsv', delimiter="\t", index_col="id")
        video_features_resnet_output_mean = video_features_resnet_output.iloc[:, :2048]
        video_features_resnet_output_max = video_features_resnet_output.iloc[:, 2048:4096]
        """
        df_index = video_features_resnet_output.index
        pca = PCA(n_components=2)
        video_features_resnet_data = pca.fit_transform(video_features_resnet_output)
        video_features_resnet = pd.DataFrame(data=video_features_resnet_data, index=df_index)
        video_features_resnet.index.name = "id"
        """
        genres["genre"] = genres["genre"].apply(ast.literal_eval)

        return tfidf_df, bert_df, genres, video_features_resnet_output_max


    if __name__ == "__main__":
        tfidf_df, bert_df, genres, video_features_resnet = initialize()
        genre_dict = get_genre_distribution(genres)
        genre_dict_frequency = {k: round(v / len(genres), 4) for k, v in genre_dict.items()}
        genre_dict_genre_frequency = {k: round(v / sum(genre_dict.values()), 4) for k, v in genre_dict.items()}
        average_genres_song = sum(genre_dict.values()) / len(genres)
        print(f"Genre: {genre_dict}\nGenre count: {len(genre_dict)}\n"
              f"Genre frequency to song count: {genre_dict_frequency}\n"
              f"Genre frequency to genre count: {genre_dict_genre_frequency}\n"
              f"Average genres per song: {average_genres_song:.4f}\n")
        performance_metrics_s_baseline(tfidf_df, bert_df, genres)
        performance_metrics_s_video(video_features_resnet, genres)

        precision_mean, mrr_mean, ndcg10_mean, ndcg100_mean, precision_mean_bl, mrr_mean_bl, ndcg10_mean_bl, ndcg100_mean_bl = performance_metrics_s(
            tfidf_df, bert_df, genres)
        print(f"Precision: {precision_mean}\n")
        print(f"MRR: {mrr_mean}\n")
        print(f"NDCG10 {ndcg10_mean}\n")
        print(f"NDCG100 {ndcg100_mean}\n")
        print(f"Baseline Metric: \n")
        print(f"Precision Baseline: {precision_mean_bl}\n")
        print(f"MRR Baseline: {mrr_mean_bl}\n")
        print(f"NDCG10 Baseline: {ndcg10_mean_bl}\n")
        print(f"NDCG100 Baseline: {ndcg100_mean_bl}\n")


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
