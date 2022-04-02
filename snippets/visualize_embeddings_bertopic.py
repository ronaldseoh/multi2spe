#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ujson as json
import numpy as np
import pandas as pd
from bertopic import BERTopic
from umap import UMAP

from spacy.lang.en import English

import tqdm


# In[2]:


# MAG metadata
with open("../scidocs/data/paper_metadata_mag_mesh.json", "r") as mag_metadata_file:
    mag_metadata = json.load(mag_metadata_file)


# In[3]:


# What keys are there in each paper?
print(mag_metadata['fedb8360a09a326f403dcca14494e1da8a5f3adc'])
print(mag_metadata['fedb8360a09a326f403dcca14494e1da8a5f3adc'].keys())


# In[4]:


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


# In[5]:


# Load the paper ids in the MAG validation set
mag_val = pd.read_csv("../scidocs/data/mag/val.csv")

mag_val_pids = set(mag_val.pid)


# In[6]:


print(mag_val)


# In[7]:


spacy_nlp = English()


# In[8]:


# Read the embeddings jsonl created with embed.py
facet_selected = 0

mag_embeddings = []
mag_docs = []
mag_labels = []

with open("save_shard11_k-3_debug_sum_embs_original-0-9+no_sum-0-1+mean-avg_word-0-05_extra_facet_alternate_layer_8_4-alternate_identity_common_random_instance_weights-v5_cross_entropy_03-30/cls.jsonl", "r") as mag_embeddings_file:
    for line in tqdm.tqdm(mag_embeddings_file):
        paper = json.loads(line)

        if paper["paper_id"] in mag_val_pids:
            emb = paper["embedding"][facet_selected]
            mag_embeddings.append(np.array(emb))

            # Filter out some stop words
            text = (mag_metadata[paper["paper_id"]].get("title") or "") + " " + (mag_metadata[paper["paper_id"]].get("abstract") or "")
            text_filtered = ""

            spacy_output = spacy_nlp(text)

            for j, token in enumerate(spacy_output):
                if not (token.is_stop or token.is_punct):
                    text_filtered += str(token) + " "

            mag_docs.append(text_filtered)
            
            mag_labels.append(mag_label_mapping[mag_val[mag_val.pid == paper["paper_id"]].iloc[0].class_label])

mag_embeddings = np.array(mag_embeddings)


# In[9]:


umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
topic_model = BERTopic(nr_topics=len(mag_label_mapping), umap_model=umap_model)


# In[10]:


topics, probs = topic_model.fit_transform(mag_docs, mag_embeddings)


# In[11]:


topic_map_fig = topic_model.visualize_topics()


# In[13]:


topic_map_fig.write_html("topic_map_facet_{}.html".format(facet_selected))


# In[14]:


topic_hierarchy_fig = topic_model.visualize_hierarchy()


# In[16]:


topic_hierarchy_fig.write_html("topic_hierarchy_facet_{}.html".format(facet_selected))


# In[17]:


topic_word_scores_fig = topic_model.visualize_barchart()


# In[19]:


topic_word_scores_fig.write_html("topic_word_scores_facet_{}.html".format(facet_selected))


# In[20]:


topic_sim_matrix_fig = topic_model.visualize_heatmap()


# In[22]:


topic_sim_matrix_fig.write_html("topic_sim_matrix_facet_{}.html".format(facet_selected))


# In[23]:


topics_per_class = topic_model.topics_per_class(mag_docs, topics, mag_labels)
topic_per_class_fig = topic_model.visualize_topics_per_class(topics_per_class, top_n_topics=len(mag_label_mapping))


# In[25]:


topic_per_class_fig.write_html("topic_per_class_facet_{}.html".format(facet_selected))


# In[ ]:




