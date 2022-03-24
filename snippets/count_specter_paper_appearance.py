import json
import pickle
import collections
import statistics

import tqdm


def generator():
    while True:
        yield


if __name__ == '__main__':

    specter_data_file_path = "/home/bseoh/my_scratch/original_data/val.pkl"
    popularity_count = collections.defaultdict(int)
    weights = {}

    with open(specter_data_file_path, 'rb') as f_in:
        unpickler = pickle.Unpickler(f_in)

        for _ in tqdm.tqdm(generator()):
            try:
                instance = unpickler.load()

                pos_paper_id = instance.fields.get('pos_paper_id').metadata
                neg_paper_id = instance.fields.get('neg_paper_id').metadata

                popularity_count[pos_paper_id] += 1
                popularity_count[neg_paper_id] += 1

            except EOFError:
                break

    avg_weights = {}

    avg_pos_weights_ids_seen = collections.defaultdict(bool)
    avg_neg_weights_ids_seen = collections.defaultdict(bool)

    avg_pos_weights_num_seen = 0
    avg_neg_weights_num_seen = 0

    avg_pos_weights = 0
    avg_neg_weights = 0

    with open(specter_data_file_path, 'rb') as f_in:
        unpickler = pickle.Unpickler(f_in)

        for _ in tqdm.tqdm(generator()):
            try:
                instance = unpickler.load()

                pos_paper_id = instance.fields.get('pos_paper_id').metadata
                neg_paper_id = instance.fields.get('neg_paper_id').metadata

                if not avg_pos_weights_ids_seen[pos_paper_id]:
                    avg_pos_weights_num_seen += 1
                    avg_pos_weights += 1 / popularity_count[pos_paper_id]
                    avg_pos_weights_ids_seen[pos_paper_id] = True

                if not avg_neg_weights_ids_seen[neg_paper_id]:
                    avg_neg_weights_num_seen += 1
                    avg_neg_weights += 1 / popularity_count[neg_paper_id]
                    avg_neg_weights_ids_seen[neg_paper_id] = True

            except EOFError:
                break

    avg_pos_weights /= avg_pos_weights_num_seen
    avg_neg_weights /= avg_neg_weights_num_seen

    avg_pos_neg_weights = (avg_pos_weights + avg_neg_weights) / 2

    avg_weights['avg_pos_weights_num_seen'] = avg_pos_weights_num_seen
    avg_weights['avg_neg_weights_num_seen'] = avg_neg_weights_num_seen
    avg_weights['avg_pos_weights'] = avg_pos_weights
    avg_weights['avg_neg_weights'] = avg_neg_weights
    avg_weights['avg_pos_neg_weights'] = avg_pos_neg_weights

    for pid in popularity_count.keys():
        weights[pid] = (1 / popularity_count[pid]) / avg_pos_neg_weights

    with open('/home/bseoh/my_scratch/original_data/val_weights.json', 'w') as weights_file:
        json.dump(weights, weights_file)

    with open('/home/bseoh/my_scratch/original_data/val_popularity_count.json', 'w') as popularity_count_file:
        json.dump(popularity_count, popularity_count_file)

    with open('/home/bseoh/my_scratch/original_data/val_weights_avg.json', 'w') as avg_weights_file:
        json.dump(avg_weights, avg_weights_file)

    print("Max occurrence:", str(max(popularity_count.values())))
    print("Min occurrence:", str(min(popularity_count.values())))
    print("Mean occurrence:", str(statistics.mean(popularity_count.values())))
    print("Median occurrence:", str(statistics.median(popularity_count.values())))
    print("avg_weights:", str(avg_weights))
