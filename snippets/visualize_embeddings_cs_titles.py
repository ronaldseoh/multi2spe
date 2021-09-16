import random

import numpy as np
import pandas as pd
import ujson as json
import sklearn.metrics
import tqdm

random.seed(413)

NUM_FACETS = 3

K = 5

if __name__ == "__main__":
    # Load the paper ids in the MAG validation set
    mag_val = pd.read_csv("scidocs/data/mag/val.csv")

    # Pick computer science papers only using the label=4
    mag_val_cs = mag_val[mag_val.class_label == 4]

    mag_val_pids = set(mag_val_cs.pid)

    # MAG metadata to get the paper titles
    mag_metadata = json.load(open("scidocs/data/paper_metadata_mag_mesh.json", "r"))

    # Read the embeddings jsonl created with embed.py
    mag_embeddings_by_facets = {}

    for f in range(NUM_FACETS):
        mag_embeddings_by_facets[f] = []

    mag_titles = []

    with open("quartermaster/save_k-3_common_8_4_identical_random_09-12/cls.jsonl", "r") as mag_embeddings_file:
        for line in tqdm.tqdm(mag_embeddings_file):
            paper = json.loads(line)

            if paper["paper_id"] in mag_val_pids:
                for f, emb in enumerate(paper["embedding"]):
                    mag_embeddings_by_facets[f].append(np.array(emb))
                mag_titles.append(mag_metadata[paper["paper_id"]]["title"])

    for f in range(NUM_FACETS):
        mag_embeddings_by_facets[f] = np.array(mag_embeddings_by_facets[f])

    cosine_similarities_by_facets = {}
    search_results_by_facets = {}

    sample_idxs = random.sample(range(len(mag_titles)), k=10)

    for f in range(NUM_FACETS):
        cosine_similarities_by_facets[f] = sklearn.metrics.pairwise.cosine_similarity(
            mag_embeddings_by_facets[f][sample_idxs], mag_embeddings_by_facets[f])

        # Closest first
        search_results_by_facets[f] = np.flip(np.argsort(cosine_similarities_by_facets[f], axis=-1), axis=-1)
    
        # Exclude the first ones in each result as that would be the query paper itself.
        search_results_by_facets[f] = search_results_by_facets[f][:, 1:]

    # Print out the titles
    for i in sample_idxs:
        print("Query paper title:", mag_titles[i])

        print()
        
        for f in range(NUM_FACETS):
            print("Facet", f)

            # Iterate through top 5 papers from the search results for this facet
            for j in range(K):
                print(mag_titles[search_results_by_facets[f][j]])
            
            print()
