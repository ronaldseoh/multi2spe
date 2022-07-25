import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import ujson as json
import ijson
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

    # cross-domain metadata to get the paper titles
    cross_domain_metadata = ijson.parse(open("20220721_shard_3_cross/metadata.json", "r"))

    # Read the embeddings jsonl created with embed.py
    mag_embeddings_by_facets = {}
    cross_domain_embeddings_by_facets = {}

    for f in range(NUM_FACETS):
        mag_embeddings_by_facets[f] = []
        cross_domain_embeddings_by_facets[f] = []

    mag_titles = []
    cross_domain_titles = []

    # Pick cross domain papers from shard 3 query papers
    cross_domain_paper_ids_temp = set()

    with open("20220721_shard_3_cross/train.txt", "r") as query_paper_ids_file:
        for pid in query_paper_ids_file.readlines():
            cross_domain_paper_ids_temp.add(pid.rstrip())

    cross_domain_paper_ids = []

    for prefix, event, value in cross_domain_metadata:
        if event == "string":
            paper_id, field = prefix.split(".")
            
            if paper_id in cross_domain_paper_ids_temp:
                if field == "title":
                    cross_domain_paper_ids.append(paper_id)
                    cross_domain_titles.append(value)
                elif field == "abstract":
                    cross_domain_titles.append(value)

    cross_domain_sample_idxs = random.sample(range(len(cross_domain_paper_ids)), k=10)
    cross_domain_sample_paper_ids = [cross_domain_paper_ids[si] for si in cross_domain_sample_idxs]

    with open("20220721_shard_3_cross/embeddings_no_sum.jsonl", "r") as cross_domain_embedding_file:
        while True:
            try:
                paper = json.loads(cross_domain_embedding_file.readline())

                if paper["paper_id"] in cross_domain_sample_paper_ids:
                    for f, emb in enumerate(paper["embedding"]):
                        cross_domain_embeddings_by_facets[f].append(np.array(emb))
            except:
                break

    mag_labels = []

    with open("quartermaster/save_shard11_U_k-3_sum_embs_original-0-9+no_sum-0-1_extra_facet_alternate_layer_8_4-alternate_identity_common_random_cross_entropy_05-04/cls_no_sum.jsonl", "r") as mag_embeddings_file:
        for line in tqdm.tqdm(mag_embeddings_file):
            paper = json.loads(line)

            if paper["paper_id"] in mag_val_pids:
                for f, emb in enumerate(paper["embedding"]):
                    mag_embeddings_by_facets[f].append(np.array(emb))
                mag_titles.append(mag_metadata[paper["paper_id"]]["title"])
                class_label = mag_val[mag_val.pid == paper["paper_id"]].iloc[0].class_label
                mag_labels.append(class_label)

    for f in range(NUM_FACETS):
        mag_embeddings_by_facets[f] = np.array(mag_embeddings_by_facets[f])
        mag_embeddings_by_facets[f] = normalize(mag_embeddings_by_facets[f], norm="l2", axis=1)

        cross_domain_embeddings_by_facets[f] = np.array(cross_domain_embeddings_by_facets[f])
        cross_domain_embeddings_by_facets[f] = normalize(cross_domain_embeddings_by_facets[f], norm="l2", axis=1)

    distances_by_facets = {}
    search_results_by_facets = {}

    for f in range(NUM_FACETS):
        distances_by_facets[f] = sklearn.metrics.pairwise.euclidean_distances(cross_domain_embeddings_by_facets[f], mag_embeddings_by_facets[f])

        # Closest first
        search_results_by_facets[f] = np.argsort(distances_by_facets[f], axis=-1)

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
