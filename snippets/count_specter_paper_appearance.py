import json
import pickle
import collections
import statistics

import tqdm


def generator():
    while True:
        yield


if __name__ == '__main__':

    specter_data_file_path = "/gypsum/scratch1/bseoh/original_data/train.pkl"
    occurrence_count = collections.defaultdict(int)

    with open(specter_data_file_path, 'rb') as f_in:
        unpickler = pickle.Unpickler(f_in)

        for _ in tqdm.tqdm(generator()):
            try:
                instance = unpickler.load()

                pos_paper_id = instance.fields.get('pos_paper_id').metadata
                neg_paper_id = instance.fields.get('neg_paper_id').metadata

                occurrence_count[pos_paper_id] += 1
                occurrence_count[neg_paper_id] += 1

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
                    avg_pos_weights += 1 / occurrence_count[pos_paper_id]
                    avg_pos_weights_ids_seen[pos_paper_id] = True

                if not avg_neg_weights_ids_seen[neg_paper_id]:
                    avg_neg_weights_num_seen += 1
                    avg_neg_weights += 1 / occurrence_count[neg_paper_id]
                    avg_neg_weights_ids_seen[neg_paper_id] = True

            except EOFError:
                break

    avg_pos_weights /= avg_pos_weights_num_seen
    avg_num_weights /= avg_pos_weights_num_seen

    avg_pos_neg_weights = (avg_pos_weights + avg_num_weights) / 2

    avg_weights['avg_pos_weights_num_seen'] = avg_pos_weights_num_seen
    avg_weights['avg_neg_weights_num_seen'] = avg_neg_weights_num_seen
    avg_weights['avg_pos_weights'] = avg_pos_weights
    avg_weights['avg_neg_weights'] = avg_neg_weights
    avg_weights['avg_pos_neg_weights'] = avg_pos_neg_weights

    with open('/gypsum/scratch1/bseoh/original_data/train_popularity_count.json', 'w') as popularity_count_file:
        json.dump(occurrence_count, popularity_count_file)

    with open('/gypsum/scratch1/bseoh/original_data/train_avg_weights.json', 'w') as avg_weights_file:
        json.dump(avg_weights, avg_weights_file)

    print("Max occurrence:", str(max(occurrence_count.values())))
    print("Min occurrence:", str(min(occurrence_count.values())))
    print("Mean occurrence:", str(statistics.mean(occurrence_count.values())))
    print("Median occurrence:", str(statistics.mean(occurrence_count.values())))
    print("avg_weights:", str(avg_weights))
