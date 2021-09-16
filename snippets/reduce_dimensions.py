import numpy as np
import pandas as pd
import ujson as json
import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import umap
import umap.plot
import tqdm


if __name__ == "__main__":
    # Load the paper ids in the MAG validation set
    mag_val = pd.read_csv("scidocs/data/mag/val.csv")

    mag_val_pids = set(mag_val.pid)

    # Read the embeddings jsonl created with embed.py
    mag_embeddings = []
    facet_labels = []

    with open("quartermaster/save_k-3_common_8_4_identical_random_09-12/cls.jsonl", "r") as mag_embeddings_file:
        for line in tqdm.tqdm(mag_embeddings_file):
            paper = json.loads(line)
            
            if paper["paper_id"] in mag_val_pids:
                for i, emb in enumerate(paper["embedding"]):
                    mag_embeddings.append(np.array(emb))
                    facet_labels.append(i)

    # Try reducing all embeddings and color code then by the facet #
    mag_embeddings = np.array(mag_embeddings)
    facet_labels = np.array(facet_labels)

    # Run UMAP
    mapper = umap.UMAP().fit(mag_embeddings)

    # Plot
    umap.plot.points(mapper, labels=facet_labels)

    # Save the plot to a file
    plt.savefig("plot.png")
