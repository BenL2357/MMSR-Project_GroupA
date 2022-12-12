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
from sklearn.preprocessing import normalize

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

        all_songs = tfidf_df.index.tolist()

        all_songs = all_songs[0:100]

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
    def performance_metrics_s_video(video_features,genres):
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

        print(f"{precision_sum/len(all_songs)}\n")
        print(f"{mrr_sum / len(all_songs)}\n")
        print(f"{ndcg_sum_10 / len(all_songs)}\n")
        print(f"{ndcg_sum_100 / len(all_songs)}\n")

    def initialize():
        tfidf_df = pd.read_csv('./resources/id_lyrics_tf-idf_mmsr (2).tsv', delimiter="\t", index_col="id")
        bert_df = pd.read_csv('./resources/id_lyrics_bert_mmsr.tsv', delimiter="\t", index_col="id")
        genres = pd.read_csv('./resources/id_genres_mmsr.tsv', delimiter="\t", index_col="id")

        """
        video_features_resnet_output = pd.read_csv('./resources/id_resnet_mmsr.tsv', delimiter="\t", index_col="id")
        df_index = video_features_resnet_output.index
        pca = PCA(n_components=2)
        video_features_resnet_data = pca.fit_transform(video_features_resnet_output)
        video_features_resnet = pd.DataFrame(data=video_features_resnet_data, index=df_index)
        video_features_resnet.index.name = "id"
        """
        genres["genre"] = genres["genre"].apply(ast.literal_eval)

        return tfidf_df, bert_df, genres, bert_df

    # endregion

    def sim_query_fast(tfidf_df, bert_df, input_query, nvals: int = 100):

        similarity = cosine_similarity(tfidf_df.loc[input_query], tfidf_df) * 0.5 + \
                     (1 / (1 + euclidean_distances(bert_df.loc[input_query], bert_df, squared=True))) * 0.5
        return similarity



    if __name__ == "__main__":
        tfidf_df, bert_df, genres, video_features_resnet = initialize()

        all_songs = tfidf_df.index.tolist()

        all_songs = all_songs[0:5000]

        results = sim_query_fast(tfidf_df, bert_df, all_songs)

        tfidf_df.sort_values('id', ascending=True)
        bert_df.sort_values('id', ascending=True)

        precision_sum = 0
        mrr_sum = 0
        ndcg_sum_10 = 0
        ndcg_sum_100 = 0

        index = 0
        for song in all_songs:

            res = pd.DataFrame(index=tfidf_df.index.tolist())
            res.index.name = "id"
            res["similarity"] = results[index]
            res.drop([song], axis=0, inplace=True)

            res_sim = res[res["similarity"] > 0.55]
            if len(res_sim) < 100:
                res_sim = res.nlargest(100, "similarity")


            # query values
            precision_sum += precision_s(genres, song, res_sim)
            mrr_sum += mrr_s(genres, song, res_sim)
            ndcg_sum_10 += nDCG_ms(res_sim, genres, song, 10)
            ndcg_sum_100 += nDCG_ms(res_sim, genres, song, 100)
            index += 1

        nsongs = len(all_songs)
        print(precision_sum / nsongs)
        print(mrr_sum / nsongs)
        print(ndcg_sum_10 / nsongs)
        print(ndcg_sum_100 / nsongs)




