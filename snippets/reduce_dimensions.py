import pandas as pd
import ujson as json
import umap
import tqdm


NUM_FACETS = 3

if __name__ == "__main__":
    # Load the paper ids in the MAG validation set
    mag_val = pd.read_csv("scidocs/data/mag/val.csv")

    # Read the embeddings jsonl created with embed.py
    mag_embeddings = {}

    with open("save_k-3_common_8_4_identical_random_09-12/cls.jsonl", "r") as mag_embeddings_file:
        for line in tqdm.tqdm(mag_embeddings_file):
            paper = json.loads(line)

            mag_embeddings[paper["paper_id"]] = paper["embedding"]

    # Perform reduction for the embeddings of each facets
