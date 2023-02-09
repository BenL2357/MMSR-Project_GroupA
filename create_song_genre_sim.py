import ast
import pandas as pd

if __name__ == "__main__":

    genres = pd.read_csv('resources/ExperimentalData/id_genres_mmsr.tsv', delimiter="\t", index_col="id").sort_values("id")
    genres["genre"] = genres["genre"].apply(ast.literal_eval)
    genres["genre"] = genres["genre"].apply(set)

    rel_genres_nr = pd.DataFrame(0, columns=["count"], index=genres.index)

    for index, cursong in genres.iterrows():

        songgenres = set(cursong["genre"])

        for idx, song in genres.iterrows():
            if len(songgenres.intersection(song["genre"])) >= 1 and idx != index:
                rel_genres_nr.loc[index]["count"] += 1

    rel_genres_nr.to_csv("./resources/ExperimentalData/genres_count.tsv", sep="\t")