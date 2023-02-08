import ast
import math

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import scipy.stats as stats

# Flag for couple of print values
METRIC_ON = False
DEBUG = True and METRIC_ON
SEED = 22031307
FOLDER_ROOT = "./resources/ExperimentalData"

def sim_query(input_query: [str], feature_vector_1, feature_vector_2=None, feature_function_mode=0):
    if feature_function_mode == 1:
        similarity = ((cosine_similarity(feature_vector_1.loc[input_query], feature_vector_1) + 1) * 0.5) * 0.5 + \
                     (1 / (1 + euclidean_distances(feature_vector_2.loc[input_query], feature_vector_2,
                                                   squared=True))) * 0.5
    elif feature_function_mode == 2 or feature_function_mode == 4:
        similarity = ((cosine_similarity(feature_vector_1.loc[input_query], feature_vector_1) * 0.5 +
                       cosine_similarity(feature_vector_2.loc[input_query], feature_vector_2) * 0.5) + 1) * 0.5
    elif feature_function_mode == 3:
        similarity = (cosine_similarity(feature_vector_1.loc[input_query], feature_vector_1) + 1)
    else:
        raise ValueError("Feature function mode is invalid!")
    return similarity


def precision_s(genres, input_query: str, results):

    input_genres = genres.loc[input_query]["genre"]

    hits = 0
    for index, row in results.iterrows():
        if len(np.intersect1d(genres.loc[index]["genre"], input_genres)) >= 1:
            hits += 1

    return hits / len(results)

def precision_s_2(genres, input_query: str, results):

    input_genres = genres.loc[input_query]["genre"]

    hits = 0
    for index in results:
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

def mrr_s_2(genres, input_query: str, results):

    input_genres = genres.loc[input_query]["genre"]

    tries = 1
    for index in results:
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

def nDCG_ms_2(results, genres, input_query: str, p: int):

    input_genres = genres.loc[input_query]["genre"]
    relevance_scores = np.zeros((p))

    iter_index = 0
    for index in results:
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


def performance_metrics_s_baseline(tfidf_df, genres):
    print("Started Baseline Metric calculation")
    precision_sum_bl = 0
    mrr_sum_bl = 0
    ndcg_sum_10_bl = 0
    ndcg_sum_100_bl = 0

    # added sample
    all_songs = tfidf_df.sample(n=10000, random_state=SEED).index.tolist()

    rand_int = 70313022

    precision_arr_sum = 0
    recall_arr_sum = 0
    for song in all_songs:
        random_results = tfidf_df.sample(n=100, random_state=rand_int)
        # baseline values
        precision_sum_bl += precision_s(genres, song, random_results)
        mrr_sum_bl += mrr_s(genres, song, random_results)
        ndcg_sum_10_bl += nDCG_ms(random_results, genres, song, 10)
        ndcg_sum_100_bl += nDCG_ms(random_results, genres, song, 100)
        rand_int += 1

        precision_arr, recall_arr = precision_recall_plot(song, genres, random_results, False)

        precision_arr_sum = precision_arr_sum + precision_arr
        recall_arr_sum = recall_arr_sum + recall_arr

    nsongs = len(all_songs)

    precision_arr_norm = precision_arr_sum / nsongs
    recall_arr_norm = recall_arr_sum / nsongs

    plot_precision_recall(precision_arr_norm, recall_arr_norm, "Baseline", ylim_min=0.0)

    print(f"Baseline Metric: \n")
    print(f"Precision Baseline: {precision_sum_bl / nsongs}\n")
    print(f"MRR Baseline: {mrr_sum_bl / nsongs}\n")
    print(f"NDCG10 Baseline: {ndcg_sum_10_bl / nsongs}\n")
    print(f"NDCG100 Baseline: {ndcg_sum_100_bl / nsongs}\n")


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


def precision_recall_plot(song, genres, results, do_cut=True):
    if do_cut:
        result_cut = results.nlargest(100, "similarity")
    else:
        result_cut = results
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

    for index in range(1, 1 + len(relevance_class)):
        sub_relevance = relevance_class[:index]
        sub_relevance_sum = sub_relevance.sum()
        precision[index] = sub_relevance_sum / len(sub_relevance)
        if relevant_sum != 0:
            recall[index] = sub_relevance_sum / relevant_sum
        else:
            recall[index] = 0

    return precision, recall

def precision_recall_plot_2(song, genres, results):

    relevance_class = np.zeros(100)
    song_genre = genres.loc[song]["genre"]

    iter_index = 0
    for index in results:
        if len(np.intersect1d(genres.loc[index]["genre"], song_genre)) >= 1:
            relevance_class[iter_index] = 1
        iter_index += 1

    relevant_sum = relevance_class.sum()
    precision = np.zeros(101)
    recall = np.zeros(101)
    precision[0] = 1
    recall[0] = 0

    for index in range(1, 1 + len(relevance_class)):
        sub_relevance = relevance_class[:index]
        sub_relevance_sum = sub_relevance.sum()
        precision[index] = sub_relevance_sum / len(sub_relevance)
        if relevant_sum != 0:
            recall[index] = sub_relevance_sum / relevant_sum
        else:
            recall[index] = 0

    return precision, recall


def percent_delta_mean(popularity_data, query_song, query_results):
    if popularity_data.loc[query_song]["popularity"] != 0.0:
        return (popularity_data.loc[query_results.index]["popularity"].mean() - popularity_data.loc[query_song][
            "popularity"]) \
               / popularity_data.loc[query_song]["popularity"]
    else:
        return 0


def percent_delta_mean_2(popularity_data, query_song, query_results):
    if popularity_data.loc[query_song]["popularity"] != 0.0:
        return (popularity_data.loc[query_results]["popularity"].mean() - popularity_data.loc[query_song][
            "popularity"]) \
               / popularity_data.loc[query_song]["popularity"]
    else:
        return 0

def hubness(count_df):
    hist = np.zeros(count_df["count"].max()+1)
    for index, row in count_df.iterrows():
        hist[row["count"]] = hist[row["count"]] + 1
    freq = np.arange(0, len(hist), 1)
    fx = freq * hist
    mean = fx.sum() / len(count_df)
    hist2 = (hist - mean) * (hist - mean) * (hist - mean)
    expected = (freq * hist2).sum() / len(count_df)
    standard_deviation = ((hist - mean) * (hist - mean)).sum() / (len(hist) - 1)
    return expected / pow(standard_deviation, 3)




def performance_metrics(feature_vector_1, feature_vector_2, feature_function_mode, genres, popularity_data,
                        n: int = 100, thv: float = 0.0):
    precision_sum = 0
    mrr_sum = 0
    ndcg_sum_10 = 0
    ndcg_sum_100 = 0

    # getting all alphanumeric indices for all songs
    all_songs = feature_vector_1.sample(n=n, random_state=SEED).index.tolist()

    index_loop = 0

    results = sim_query(all_songs, feature_vector_1, feature_vector_2, feature_function_mode)
    precision_arr_sum = 0
    recall_arr_sum = 0
    delta_mean_array = np.zeros(n)
    for song in all_songs:
        res = pd.DataFrame(index=feature_vector_1.index.tolist())
        res.index.name = "id"
        res["similarity"] = results[index_loop]
        res.drop([song], axis=0, inplace=True)

        if thv != 0.0:
            res_sim = res[res["similarity"] > thv]
            if len(res_sim) < 100:
                res_sim = res.nlargest(100, "similarity")
            else:
                res_sim = res_sim.sort_values(by=["similarity"], ascending=False)
        else:
            res_sim = res.nlargest(100, "similarity")
        if METRIC_ON:
            delta_mean_array[index_loop] = percent_delta_mean(popularity_data, song, res_sim)

            precision_arr, recall_arr = precision_recall_plot(song, genres, res_sim)

            precision_arr_sum = precision_arr_sum + precision_arr
            recall_arr_sum = recall_arr_sum + recall_arr

            precision_sum += precision_s(genres, song, res_sim)
            mrr_sum += mrr_s(genres, song, res_sim)
            ndcg_sum_10 += nDCG_ms(res_sim, genres, song, 10)
            ndcg_sum_100 += nDCG_ms(res_sim, genres, song, 100)
            index_loop += 1
    if METRIC_ON:
        precision_arr_norm = precision_arr_sum / len(all_songs)
        recall_arr_norm = recall_arr_sum / len(all_songs)

        if feature_function_mode == 1:
            metric_name = "Lyric Features: TF-IDF, BERT"
        elif feature_function_mode == 2:
            metric_name = "Video Features: VGG19"
        elif feature_function_mode == 3:
            metric_name = "Audio Feature: MFCC BoW"
        elif feature_function_mode == 4:
            metric_name = "Audio BL Features: Spectral, Spectral Contrast"
        else:
            metric_name = "Undefinied"

        plot_precision_recall(precision_arr_norm, recall_arr_norm, metric_name)

    if DEBUG:
        print(f"Performan Metrics for {metric_name}")
        print(f"Precision: {precision_sum / len(all_songs)}\n")
        print(f"MRR: {mrr_sum / len(all_songs)}\n")
        print(f"NDCG10: {ndcg_sum_10 / len(all_songs)}\n")
        print(f"NDCG100: {ndcg_sum_100 / len(all_songs)}\n")
        print(f"Median Delta Mean: {np.median(delta_mean_array)}")

    return precision_sum / len(all_songs), mrr_sum / len(all_songs), ndcg_sum_10 / len(
        all_songs), ndcg_sum_100 / len(all_songs)

def merged_performance_metrics(tfidf_df, bert_df, genres, video_features_resnet_max, video_features_resnet_mean, mfcc_bow, spectral, spectral_contrast, spotify_data):
    all_results = pd.DataFrame(index=genres.index, columns=(["results"]))
    all_results.index.name = "id"
    count_df = pd.DataFrame(index=genres.index, columns=(["count"])).fillna(0)
    all_results.index.name = "id"

    splitted_dataset = np.array_split(genres, 3)
    iter_counter = 0
    for dataset in splitted_dataset:
        query_songs = dataset.index
        results_lyrics = sim_query(query_songs, tfidf_df, bert_df, 1)
        results_video = sim_query(query_songs, video_features_resnet_max, video_features_resnet_mean, 2)
        results_audio_bow = sim_query(query_songs, mfcc_bow, None, 3)
        results_audio_spectral = sim_query(query_songs, spectral, spectral_contrast, 4)

        results = results_lyrics * 0.25 + results_video * 0.25 + results_audio_bow * 0.25 + results_audio_spectral * 0.25
        for i in range(len(query_songs)):
            results[i, i+iter_counter] = 0

        res = pd.DataFrame(results.T, index=genres.index, columns=query_songs)
        res.index.name = "id"

        for col in res.columns:
            res_largest = res[col].nlargest(100)
            for index in res_largest.index.tolist():
                count_df.loc[index]["count"] = count_df.loc[index]["count"] + 1
            all_results.loc[col]["results"] = res_largest.index.tolist()
        iter_counter += len(dataset)

    for key, value in all_results.iterrows():
        if key in value["results"]:
            print(key)

    index_loop = 0
    delta_mean_array = np.zeros(len(genres))
    precision_sum = 0
    mrr_sum = 0
    ndcg_sum_10 = 0
    ndcg_sum_100 = 0

    precision_arr_sum = 0
    recall_arr_sum = 0

    all_results.dropna(inplace=True)

    hubness_100 = hubness(count_df)

    all_results.to_csv("./resources/results.csv")

    for index, result in all_results.iterrows():
        if not result.empty or not pd.isna(result):
            delta_mean_array[index_loop] = percent_delta_mean_2(spotify_data, index, result["results"])

            precision_arr, recall_arr = precision_recall_plot_2(index, genres, result["results"])

            precision_arr_sum = precision_arr_sum + precision_arr
            recall_arr_sum = recall_arr_sum + recall_arr

            precision_sum += precision_s_2(genres, index, result["results"])
            mrr_sum += mrr_s_2(genres, index, result["results"])
            ndcg_sum_10 += nDCG_ms_2(result["results"], genres, index, 10)
            ndcg_sum_100 += nDCG_ms_2(result["results"], genres, index, 100)
            index_loop += 1


    precision_arr_norm = precision_arr_sum / len(genres)
    recall_arr_norm = recall_arr_sum / len(genres)

    plot_precision_recall(precision_arr_norm, recall_arr_norm, "All songs")



    print(f"Performan Metrics for {len(genres)} songs")
    print(f"Precision: {precision_sum / len(genres)}\n")
    print(f"MRR: {mrr_sum / len(genres)}\n")
    print(f"NDCG10: {ndcg_sum_10 / len(genres)}\n")
    print(f"NDCG100: {ndcg_sum_100 / len(genres)}\n")
    print(f"Median Delta Mean: {np.median(delta_mean_array)}")
    print(f"Hubness k=100: {hubness_100}")





def kendall_tau_correlation_calculation(tfidf_df, bert_df, video_features_resnet_max, video_features_resnet_mean,
                                        mfcc_bow, spectral, spectral_contrast, song_count=1000):
    all_songs = tfidf_df.sample(n=song_count, random_state=SEED).index.tolist()

    results = list()
    results.append(sim_query(all_songs, tfidf_df, bert_df, 1))
    results.append(sim_query(all_songs, video_features_resnet_mean, video_features_resnet_max, 2))
    results.append(sim_query(all_songs, mfcc_bow, None, 3))
    results.append(sim_query(all_songs, spectral, spectral_contrast, 4))
    song_index = 0
    correclation_matrix = np.zeros(shape=(len(results), len(results)))

    for song in all_songs:
        for index, item in enumerate(results):
            res = pd.DataFrame(index=tfidf_df.index.tolist())
            res.index.name = "id"
            res["similarity"] = results[index][song_index]
            res.drop([song], axis=0, inplace=True)

            res_modality_1 = res.nlargest(100, "similarity")

            for index2, item2 in enumerate(results):
                if index != index2:
                    res = pd.DataFrame(index=tfidf_df.index.tolist())
                    res.index.name = "id"
                    res["similarity"] = results[index2][song_index]
                    res.drop([song], axis=0, inplace=True)

                    res_modality_2 = res.nlargest(100, "similarity")

                    idx1 = pd.Index(res_modality_1.index)
                    idx2 = pd.Index(res_modality_2.index)

                    intersection_result = idx1.intersection(idx2)

                    rankings_modal1 = np.zeros(len(intersection_result))
                    rankings_modal2 = np.zeros(len(intersection_result))
                    for index3, element in enumerate(intersection_result):
                        rankings_modal1[index3] = idx1.get_loc(element) + 1
                        rankings_modal2[index3] = idx2.get_loc(element) + 1
                    if len(intersection_result) > 1:
                        correlation_value, _ = stats.kendalltau(rankings_modal1, rankings_modal2)
                        correclation_matrix[index, index2] = correclation_matrix[index, index2] + correlation_value
        song_index += 1

    correclation_matrix = correclation_matrix / len(all_songs)
    print(correclation_matrix)


def initialize():
    tfidf_df = pd.read_csv(f'{FOLDER_ROOT}/id_lyrics_tf-idf_mmsr.tsv', delimiter="\t", index_col="id").sort_values(
        "id")
    bert_df = pd.read_csv(f'{FOLDER_ROOT}/id_lyrics_bert_mmsr.tsv', delimiter="\t", index_col="id").sort_values("id")
    genres = pd.read_csv(f'{FOLDER_ROOT}/id_genres_mmsr.tsv', delimiter="\t", index_col="id").sort_values("id")

    video_features_resnet_output = pd.read_csv(f'{FOLDER_ROOT}/id_resnet_mmsr.tsv', delimiter="\t",
                                               index_col="id").sort_values("id")
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

    mfcc_bow = pd.read_csv(f'{FOLDER_ROOT}/id_mfcc_bow_mmsr.tsv', delimiter="\t", index_col="id").sort_values("id")

    spectral = pd.read_csv(f'{FOLDER_ROOT}/id_blf_spectral_mmsr.tsv', delimiter="\t", index_col="id").sort_values("id")
    spectral_contrast = pd.read_csv(f'{FOLDER_ROOT}/id_blf_spectralcontrast_mmsr.tsv', delimiter="\t",
                                    index_col="id").sort_values("id")
    spotify_data = pd.read_csv(f'{FOLDER_ROOT}/id_metadata_mmsr.tsv', delimiter="\t", index_col="id").sort_values("id")

    genres["genre"] = genres["genre"].apply(ast.literal_eval)

    return tfidf_df, bert_df, genres, video_features_resnet_max, video_features_resnet_mean, mfcc_bow, spectral, spectral_contrast, spotify_data


if __name__ == "__main__":
    tfidf_df, bert_df, genres, video_features_resnet_max, video_features_resnet_mean, mfcc_bow, spectral, spectral_contrast, spotify_data = initialize()
    genre_dict = get_genre_distribution(genres)
    genre_dict_frequency = {k: round(v / len(genres), 4) for k, v in genre_dict.items()}
    genre_dict_genre_frequency = {k: round(v / sum(genre_dict.values()), 4) for k, v in genre_dict.items()}
    average_genres_song = sum(genre_dict.values()) / len(genres)
    print(f"Genre: {genre_dict}\nGenre count: {len(genre_dict)}\n"
          f"Genre frequency to song count: {genre_dict_frequency}\n"
          f"Genre frequency to genre count: {genre_dict_genre_frequency}\n"
          f"Average genres per song: {average_genres_song:.4f}\n")
    # performance_metrics_s_baseline(tfidf_df, genres)
    #performance_metrics(tfidf_df, bert_df, 1, genres, spotify_data, 10000, thv=0.55)
    #performance_metrics(video_features_resnet_max, video_features_resnet_mean, 2, genres, spotify_data, 10000)
    #performance_metrics(mfcc_bow, None, 3, genres, spotify_data, 10000)
    #performance_metrics(spectral, spectral_contrast, 4, genres, spotify_data, 10000)
    merged_performance_metrics(tfidf_df, bert_df, genres, video_features_resnet_max, video_features_resnet_mean, mfcc_bow, spectral, spectral_contrast, spotify_data)
    kendall_tau_correlation_calculation(tfidf_df, bert_df, video_features_resnet_max, video_features_resnet_mean,
                                        mfcc_bow, spectral, spectral_contrast)
