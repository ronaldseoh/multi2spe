import json
import pickle
#import collections
#import statistics

import tqdm


def generator():
    while True:
        yield


if __name__ == '__main__':

    specter_data_file_path = "/home/bseoh/my_scratch/20220327_shard_11/preprocessed/data-train.p"
    mapping = {}

    with open(specter_data_file_path, 'rb') as f_in:
        unpickler = pickle.Unpickler(f_in)

        for _ in tqdm.tqdm(generator()):
            try:
                instance = unpickler.load()

                source_paper_id = instance.fields.get('source_paper_id').metadata
                pos_paper_id = instance.fields.get('pos_paper_id').metadata
                neg_paper_id = instance.fields.get('neg_paper_id').metadata
            except EOFError:
                break

            mapping[source_paper_id] = source_paper_id
            mapping[pos_paper_id] = pos_paper_id
            mapping[neg_paper_id] = neg_paper_id

    with open('/home/bseoh/my_scratch/20220327_shard_11/preprocessed/train_all_paper_ids.json', 'w') as all_paper_ids_file:
        json.dump(mapping, all_paper_ids_file)
