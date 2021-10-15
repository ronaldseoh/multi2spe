import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import ujson as json
import sklearn.metrics
import tqdm

random.seed(413)

NUM_FACETS = 3

K = 5

mag_label_mapping = {
    0: "Art",
    1: "Biology",
    2: "Business",
    3: "Chemistry",
    4: "Computer science",
    5: "Economics",
    6: "Engineering",
    7: "Environmental science",
    8: "Geography",
    9: "Geology",
    10:	"History",
    11: "Materials science",
    12: "Mathematics",
    13:	"Medicine",
    14: "Philosophy",
    15: "Physics",
    16: "Political science",
    17: "Psychology",
    18: "Sociology",
}

if __name__ == "__main__":
    # Load the paper ids in the MAG validation set
    mag_val = pd.read_csv("scidocs/data/mag/val.csv")
    mag_val_pids = set(mag_val.pid)

    # MAG metadata to get the paper titles
    mag_metadata = json.load(open("scidocs/data/paper_metadata_mag_mesh.json", "r"))

    # Read the embeddings jsonl created with embed.py
    mag_embeddings_by_facets = {}

    for f in range(NUM_FACETS):
        mag_embeddings_by_facets[f] = []

    mag_titles = []
    mag_labels = []

    # Pick computer science papers only using the label=4
    mag_val_cs_indexes = []

    with open("quartermaster/save_k-3_common_layer_8_4_identity_nsp_cross_entropy_10-06/cls.jsonl", "r") as mag_embeddings_file:
        for line in tqdm.tqdm(mag_embeddings_file):
            paper = json.loads(line)

            if paper["paper_id"] in mag_val_pids:
                for f, emb in enumerate(paper["embedding"]):
                    mag_embeddings_by_facets[f].append(np.array(emb))
                mag_titles.append(mag_metadata[paper["paper_id"]]["title"])
                class_label = mag_val[mag_val.pid == paper["paper_id"]].iloc[0].class_label
                mag_labels.append(class_label)
    
                if class_label == 4:
                    mag_val_cs_indexes.append(len(mag_labels) - 1)

    for f in range(NUM_FACETS):
        mag_embeddings_by_facets[f] = np.array(mag_embeddings_by_facets[f])
        mag_embeddings_by_facets[f] = normalize(mag_embeddings_by_facets[f], norm="l2", axis=1)

    distances_by_facets = {}
    search_results_by_facets = {}

    sample_idxs = random.sample(mag_val_cs_indexes, k=10)

    for f in range(NUM_FACETS):
        distances_by_facets[f] = sklearn.metrics.pairwise.euclidean_distances(
            mag_embeddings_by_facets[f][sample_idxs], mag_embeddings_by_facets[f])

        # Closest first
        search_results_by_facets[f] = np.argsort(distances_by_facets[f], axis=-1)

        # Exclude the first ones in each result as that would be the query paper itself.
        search_results_by_facets[f] = search_results_by_facets[f][:, 1:]

    # Write down the titles
    with open("titles.txt", "w") as titles_file:
        for i in range(len(sample_idxs)):
            titles_file.write("Query paper " + str(i) + " title: " + mag_titles[sample_idxs[i]] + "\n" + "\n")
            
            for f in range(NUM_FACETS):
                titles_file.write("Facet " + str(f) + "\n" + "\n")

                # Iterate through top 5 papers from the search results for this facet
                for j in range(K):
                    titles_file.write(mag_titles[search_results_by_facets[f][i][j]])
                    titles_file.write(" (" + mag_label_mapping[mag_labels[search_results_by_facets[f][i][j]]] + ")")
                    titles_file.write("\n")

                titles_file.write("\n")

            titles_file.write("----------------------------------------------------------------------------------\n\n")
