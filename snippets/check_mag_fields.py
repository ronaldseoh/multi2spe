import json
import pickle
import collections
import statistics

import tqdm


def generator():
    while True:
        yield


if __name__ == '__main__':

    query_paper_count_by_mag_field = collections.defaultdict(int)
    pos_paper_count_by_mag_field = collections.defaultdict(int)
    neg_paper_count_by_mag_field = collections.defaultdict(int)

    num_query_paper_ids_found_mag = 0
    num_pos_paper_ids_found_mag = 0
    num_neg_paper_ids_found_mag = 0

    unique_paper_ids = set()

    extra_metadata = json.load(open('/home/bseoh/my_scratch/20220327_shard_11/preprocessed/train_extra_metadata.json', 'r'))

    with open("/home/bseoh/my_scratch/20220327_shard_11/preprocessed/data-train.p", 'rb') as f_in:
        unpickler = pickle.Unpickler(f_in)
        for _ in tqdm.tqdm(generator()):
            try:
                instance = unpickler.load()

                query_paper_id = instance.fields.get('source_paper_id').metadata
                pos_paper_id = instance.fields.get('pos_paper_id').metadata
                neg_paper_id = instance.fields.get('neg_paper_id').metadata

                unique_paper_ids.add(query_paper_id)
                unique_paper_ids.add(pos_paper_id)
                unique_paper_ids.add(neg_paper_id)

                if query_paper_id in extra_metadata.keys() and extra_metadata[query_paper_id]['mag_field_of_study'] is not None:
                    num_query_paper_ids_found_mag += 1

                    for f in extra_metadata[query_paper_id]['mag_field_of_study']:
                        query_paper_count_by_mag_field[f] += 1

                if pos_paper_id in extra_metadata.keys() and extra_metadata[pos_paper_id]['mag_field_of_study'] is not None:
                    num_pos_paper_ids_found_mag += 1

                    for f in extra_metadata[pos_paper_id]['mag_field_of_study']:
                        pos_paper_count_by_mag_field[f] += 1

                if neg_paper_id in extra_metadata.keys() and extra_metadata[neg_paper_id]['mag_field_of_study'] is not None:
                    num_neg_paper_ids_found_mag += 1

                    for f in extra_metadata[neg_paper_id]['mag_field_of_study']:
                        neg_paper_count_by_mag_field[f] += 1

            except EOFError:
                break
