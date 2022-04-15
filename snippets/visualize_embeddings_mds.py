"""
UMAP: https://github.com/lmcinnes/umap/blob/master/LICENSE.txt

BSD 3-Clause License

Copyright (c) 2017, Leland McInnes
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import pandas as pd
import ujson as json
import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.manifold import MDS, TSNE
import umap
import umap.plot
import tqdm


# originally umap.plot.points()
def plot(
    points,
    labels=None,
    values=None,
    theme=None,
    cmap="Blues",
    color_key=None,
    color_key_cmap="Spectral",
    background="white",
    width=800,
    height=800,
    show_legend=True,
    subset_points=None,
    ax=None,
    alpha=None,
):

    if theme is not None:
        cmap = umap.plot._themes[theme]["cmap"]
        color_key_cmap = umap.plot._themes[theme]["color_key_cmap"]
        background = umap.plot._themes[theme]["background"]

    if labels is not None and values is not None:
        raise ValueError(
            "Conflicting options; only one of labels or values should be set"
        )

    if alpha is not None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha must be between 0 and 1 inclusive")

    if subset_points is not None:
        if len(subset_points) != points.shape[0]:
            raise ValueError(
                "Size of subset points ({}) does not match number of input points ({})".format(
                    len(subset_points), points.shape[0]
                )
            )
        points = points[subset_points]

        if labels is not None:
            labels = labels[subset_points]
        if values is not None:
            values = values[subset_points]

    if points.shape[1] != 2:
        raise ValueError("Plotting is currently only implemented for 2D embeddings")

    font_color = umap.plot._select_font_color(background)

    if ax is None:
        dpi = plt.rcParams["figure.dpi"]
        fig = plt.figure(figsize=(width / dpi, height / dpi))
        ax = fig.add_subplot(111)

    if points.shape[0] <= width * height // 10:
        ax = umap.plot._matplotlib_points(
            points,
            ax,
            labels,
            values,
            cmap,
            color_key,
            color_key_cmap,
            background,
            width,
            height,
            show_legend,
        )
    else:
        # Datashader uses 0-255 as the range for alpha, with 255 as the default
        if alpha is not None:
            alpha = alpha * 255
        else:
            alpha = 255

        ax = umap.plot._datashade_points(
            points,
            ax,
            labels,
            values,
            cmap,
            color_key,
            color_key_cmap,
            background,
            width,
            height,
            show_legend,
            alpha,
        )

    ax.set(xticks=[], yticks=[])

    ax.text(
        0.99,
        0.01,
        "TSNE",
        transform=ax.transAxes,
        horizontalalignment="right",
        color=font_color,
    )

    return ax

if __name__ == "__main__":
    # Load the paper ids in the MAG validation set
    mag_val = pd.read_csv("scidocs/data/mag/val.csv")

    mag_val_pids = set(mag_val.pid)

    # Read the embeddings jsonl created with embed.py
    mag_embeddings = []
    facet_labels = []
    mag_labels = []

    with open("quartermaster/save_shard11_k-3_debug_sum_embs_original-0-9+no_sum-0-1+mean-avg_word-0-05_extra_facet_alternate_layer_8_4-alternate_identity_common_random_cross_entropy_03-30/cls_no_sum.jsonl", "r") as mag_embeddings_file:
        for line in tqdm.tqdm(mag_embeddings_file):
            paper = json.loads(line)
            
            if paper["paper_id"] in mag_val_pids:
                embs_paper = []

                for i, emb in enumerate(paper["embedding"]):
                    embs_paper.append(np.array(emb))

                emb_mean = np.mean(embs_paper, axis=0)

                for i, emb in enumerate(embs_paper):
                    mag_embeddings.append(emb - emb_mean)
                    facet_labels.append(i)
                    mag_labels.append(mag_val[mag_val.pid == paper["paper_id"]].iloc[0].class_label)

    # Try reducing all embeddings and color code then by the facet #
    mag_embeddings = np.array(mag_embeddings)
    mag_embeddings = normalize(mag_embeddings, norm="l2", axis=1)
    facet_labels = np.array(facet_labels)
    mag_labels = np.array(mag_labels)

    # Run MDS
    # mds = MDS(2, random_state=0)
    # mag_embeddings_2d = mds.fit_transform(mag_embeddings)
    tsne = TSNE(n_components=2, random_state=0, init='random')
    mag_embeddings_2d = tsne.fit_transform(mag_embeddings)

    # Plot first with facet #
    plot(mag_embeddings_2d, labels=facet_labels)

    # Save the plot to a file
    plt.savefig("plot1.png")
    
    # Then plot with MAG labels
    plot(mag_embeddings_2d, labels=mag_labels)    

    # Save the plot to a file
    plt.savefig("plot2.png")
