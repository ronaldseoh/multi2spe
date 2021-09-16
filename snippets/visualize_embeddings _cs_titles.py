import random

import numpy as np
import pandas as pd
import ujson as json
import sklearn.metrics
import tqdm

random.seed(413)

NUM_FACETS = 3

if __name__ == "__main__":
    # Load the paper ids in the MAG validation set
    mag_val = pd.read_csv("scidocs/data/mag/val.csv")

    # Pick computer science papers only using the label=4
    mag_val_cs = mag_val[mag_val.class_label == 4]

    mag_val_pids = set(mag_val_cs.pid)

    mag_val_pids_samples = random.sample(mag_val_pids, k=10)

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

            if paper["paper_id"] in mag_val_pids_samples:
                for f, emb in enumerate(paper["embedding"]):
                    mag_embeddings_by_facets[f].append(np.array(emb))
                mag_titles.append(mag_metadata[paper["paper_id"]]["title"])

    cosine_similarities_by_facets = {}

    for f in range(NUM_FACETS):
        cosine_similarities_by_facets[f] = sklearn.metrics.pairwise.cosine_similarity(mag_embeddings_by_facets[f])
