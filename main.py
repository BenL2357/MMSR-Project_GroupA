import ast

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

# Flag for couple of print values
DEBUG = True
SEED = 22031307

if __name__ == "__main__":

    def sim_query(tfidf_df, bert_df, input_query: [str]):
        similarity = cosine_similarity(tfidf_df.loc[input_query], tfidf_df) * 0.5 + \
                     (1 / (1 + euclidean_distances(bert_df.loc[input_query], bert_df, squared=True))) * 0.5
        return similarity


    def sim_query_s_video(video_feature_vector_1, video_feature_vector_2, input_query: [str]):
        similarity = cosine_similarity(video_feature_vector_1.loc[input_query], video_feature_vector_1) * 0.5 + \
                     cosine_similarity(video_feature_vector_2.loc[input_query], video_feature_vector_2) * 0.5
        return similarity

    def sim_query_s_mfcc(mfcc_bow, input_query: [str]):
        similarity = cosine_similarity(mfcc_bow.loc[input_query], mfcc_bow)
        return similarity

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


    def performance_metrics_s(tfidf_df, bert_df, genres, n: int = 100, thv: float = 0.55):
        precision_sum = 0
        mrr_sum = 0
        ndcg_sum_10 = 0
        ndcg_sum_100 = 0

        # added sample
        all_songs = tfidf_df.sample(n=n, random_state=SEED).index.tolist()

        results = sim_query(tfidf_df, bert_df, all_songs)

        index = 0
        for song in all_songs:
            # creating a data frame with the result for the specific song
            res = pd.DataFrame(index=tfidf_df.index.tolist())
            res.index.name = "id"
            res["similarity"] = results[index]
            res.drop([song], axis=0, inplace=True)

            res_sim = res[res["similarity"] > thv]
            if len(res_sim) < 100:
                res_sim = res.nlargest(100, "similarity")
            else:
                res_sim = res_sim.sort_values(by=["similarity"], ascending=False)

            # query values
            precision_sum += precision_s(genres, song, res_sim)
            mrr_sum += mrr_s(genres, song, res_sim)
            ndcg_sum_10 += nDCG_ms(res_sim, genres, song, 10)
            ndcg_sum_100 += nDCG_ms(res_sim, genres, song, 100)
            index += 1

        nsongs = len(all_songs)

        if DEBUG:
            print(f"Precision: {precision_sum / nsongs}\n")
            print(f"MRR: {mrr_sum / nsongs}\n")
            print(f"NDCG10 {ndcg_sum_10 / nsongs}\n")
            print(f"NDCG100 {ndcg_sum_100 / nsongs}\n")

        return precision_sum / nsongs, mrr_sum / nsongs, ndcg_sum_10 / nsongs, ndcg_sum_100 / nsongs


    def performance_metrics_s_baseline(tfidf_df, genres):
        print("Started Baseline Metric calculation")
        precision_sum_bl = 0
        mrr_sum_bl = 0
        ndcg_sum_10_bl = 0
        ndcg_sum_100_bl = 0

        # added sample
        all_songs = tfidf_df.sample(n=len(tfidf_df),random_state=SEED).index.tolist()

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


    def performance_metrics_s_video(video_features_max, video_features_mean, genres, n: int = 100, thv: float = 0.55):
        precision_sum = 0
        mrr_sum = 0
        ndcg_sum_10 = 0
        ndcg_sum_100 = 0

        all_songs = video_features_max.sample(n=n, random_state=SEED).index.tolist()

        results = sim_query_s_video(video_features_max, video_features_mean, all_songs)

        index = 0
        for song in all_songs:
            res = pd.DataFrame(index=video_features_max.index.tolist())
            res.index.name = "id"
            res["similarity"] = results[index]
            res.drop([song], axis=0, inplace=True)

            res_sim = res.nlargest(100, "similarity")

            precision_sum += precision_s(genres, song, res_sim)
            mrr_sum += mrr_s(genres, song, res_sim)
            ndcg_sum_10 += nDCG_ms(res_sim, genres, song, 10)
            ndcg_sum_100 += nDCG_ms(res_sim, genres, song, 100)
            index += 1

        print(f"Video Metric:\n")
        print(f"{precision_sum / len(all_songs)}\n")
        print(f"{mrr_sum / len(all_songs)}\n")
        print(f"{ndcg_sum_10 / len(all_songs)}\n")
        print(f"{ndcg_sum_100 / len(all_songs)}\n")

    def performance_metrics_s_spectral(spectral, spectral_contrast, genres, n: int = 100, thv: float = 0.55):
        precision_sum = 0
        mrr_sum = 0
        ndcg_sum_10 = 0
        ndcg_sum_100 = 0

        all_songs = spectral.sample(n=n, random_state=SEED).index.tolist()

        results = sim_query_s_video(spectral, spectral_contrast, all_songs)
        precision_arr_sum = np.zeros(101)
        recall_arr_sum = np.zeros(101)

        index = 0
        for song in all_songs:
            res = pd.DataFrame(index=spectral.index.tolist())
            res.index.name = "id"
            res["similarity"] = results[index]
            res.drop([song], axis=0, inplace=True)

            res_sim = res.nlargest(100, "similarity")

            precision_arr, recall_arr = precision_recall_plot(song, genres, res_sim)

            precision_arr_sum = precision_arr_sum + precision_arr
            recall_arr_sum = recall_arr_sum + recall_arr

            precision_sum += precision_s(genres, song, res_sim)
            mrr_sum += mrr_s(genres, song, res_sim)
            ndcg_sum_10 += nDCG_ms(res_sim, genres, song, 10)
            ndcg_sum_100 += nDCG_ms(res_sim, genres, song, 100)
            index += 1

        precision_to_plot = precision_arr_sum / len(all_songs)
        recall_to_plot = recall_arr_sum / len(all_songs)

        plot_precision_recall(precision_to_plot, recall_to_plot, "Spectral")

        print(f"Spectral Metrics:\n")
        print(f"{precision_sum / len(all_songs)}\n")
        print(f"{mrr_sum / len(all_songs)}\n")
        print(f"{ndcg_sum_10 / len(all_songs)}\n")
        print(f"{ndcg_sum_100 / len(all_songs)}\n")

    def plot_precision_recall(precision_to_plot, recall_to_plot, label, ylim_min=0.5):
        # create precision recall curve
        fig, ax = plt.subplots()
        ax.plot(recall_to_plot, precision_to_plot, color='purple')
        ax.set_ylim([ylim_min, 1.03])

        # add axis labels to plot
        ax.set_title(f'Precision-Recall Curve {label}')
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')

        plt.show()

    def precision_recall_plot(song, genres, results):
        result_cut = results.nlargest(100, "similarity")
        relevance_class = np.zeros(100)
        song_genre = genres.loc[song]["genre"]

        iter_index = 0
        for index, row in result_cut.iterrows():
            if len(np.intersect1d(genres.loc[index]["genre"], song_genre)) >= 1:
                relevance_class[iter_index] = 1
            iter_index += 1

        relevant_sum = relevance_class.sum()
        precision = np.zeros(101)
        recall = np.zeros(101)
        precision[0] = 1
        recall[0] = 0

        for index in range(1, 1+len(relevance_class)):
            sub_relevance = relevance_class[:index]
            sub_relevance_sum = sub_relevance.sum()
            precision[index] = sub_relevance_sum/len(sub_relevance)
            if relevant_sum != 0:
                recall[index] = sub_relevance_sum/relevant_sum
            else:
                recall[index] = 0

        return precision, recall




    def performance_metrics_s_mfcc(mfcc_bow, genres, n: int = 100, thv: float = 0.55):
        precision_sum = 0
        mrr_sum = 0
        ndcg_sum_10 = 0
        ndcg_sum_100 = 0

        all_songs = mfcc_bow.sample(n=n, random_state=SEED).index.tolist()

        results = sim_query_s_mfcc(mfcc_bow, all_songs)

        index = 0
        for song in all_songs:
            res = pd.DataFrame(index=mfcc_bow.index.tolist())
            res.index.name = "id"
            res["similarity"] = results[index]
            res.drop([song], axis=0, inplace=True)

            res_sim = res.nlargest(100, "similarity")

            precision_sum += precision_s(genres, song, res_sim)
            mrr_sum += mrr_s(genres, song, res_sim)
            ndcg_sum_10 += nDCG_ms(res_sim, genres, song, 10)
            ndcg_sum_100 += nDCG_ms(res_sim, genres, song, 100)
            index += 1

        print(f"MFCC BoAW Metric:\n")
        print(f"{precision_sum / len(all_songs)}\n")
        print(f"{mrr_sum / len(all_songs)}\n")
        print(f"{ndcg_sum_10 / len(all_songs)}\n")
        print(f"{ndcg_sum_100 / len(all_songs)}\n")


    def initialize():
        #TODO maybe split this function for modalities
        tfidf_df = pd.read_csv('./resources/id_lyrics_tf-idf_mmsr.tsv', delimiter="\t", index_col="id")
        bert_df = pd.read_csv('./resources/id_lyrics_bert_mmsr.tsv', delimiter="\t", index_col="id")
        genres = pd.read_csv('./resources/id_genres_mmsr.tsv', delimiter="\t", index_col="id")
        """
        video_features_resnet_output = pd.read_csv('./resources/id_resnet_mmsr.tsv', delimiter="\t", index_col="id")
        video_features_resnet_output_mean = video_features_resnet_output.iloc[:, :2048]
        video_features_resnet_output_max = video_features_resnet_output.iloc[:, 2048:4096]

        df_index = video_features_resnet_output_mean.index
        pca = PCA(n_components=512, random_state=SEED)
        video_features_resnet_data_mean = pca.fit_transform(video_features_resnet_output_mean)
        video_features_resnet_data_max = pca.fit_transform(video_features_resnet_output_max)
        video_features_resnet_mean = pd.DataFrame(data=video_features_resnet_data_mean, index=df_index)
        video_features_resnet_mean.index.name = "id"
        video_features_resnet_max = pd.DataFrame(data=video_features_resnet_data_max, index=df_index)
        video_features_resnet_max.index.name = "id"
        """
        mfcc_bow = pd.read_csv('./resources/id_mfcc_bow_mmsr.tsv', delimiter="\t", index_col="id")

        spectral = pd.read_csv('./resources/id_blf_spectral_mmsr.tsv', delimiter="\t", index_col="id")
        spectral_contrast = pd.read_csv('./resources/id_blf_spectralcontrast_mmsr.tsv', delimiter="\t", index_col="id")
        video_features_resnet_max = 0
        video_features_resnet_mean = 0

        genres["genre"] = genres["genre"].apply(ast.literal_eval)

        return tfidf_df, bert_df, genres, video_features_resnet_max, video_features_resnet_mean, mfcc_bow, spectral, spectral_contrast


    if __name__ == "__main__":
        tfidf_df, bert_df, genres, video_features_resnet_max, video_features_resnet_mean, mfcc_bow, spectral, spectral_contrast = initialize()
        # genre_dict = get_genre_distribution(genres)
        # genre_dict_frequency = {k: round(v / len(genres), 4) for k, v in genre_dict.items()}
        # genre_dict_genre_frequency = {k: round(v / sum(genre_dict.values()), 4) for k, v in genre_dict.items()}
        # average_genres_song = sum(genre_dict.values()) / len(genres)
        # print(f"Genre: {genre_dict}\nGenre count: {len(genre_dict)}\n"
        #      f"Genre frequency to song count: {genre_dict_frequency}\n"
        #      f"Genre frequency to genre count: {genre_dict_genre_frequency}\n"
        #      f"Average genres per song: {average_genres_song:.4f}\n")
        #performance_metrics_s_baseline(tfidf_df, genres)
        performance_metrics_s_spectral(spectral, spectral_contrast, genres, 100)
        performance_metrics_s_video(video_features_resnet_max, video_features_resnet_mean, genres, 100)
        performance_metrics_s_mfcc(mfcc_bow, genres, n=100)
        performance_metrics_s(tfidf_df, bert_df, genres,  n=100)
