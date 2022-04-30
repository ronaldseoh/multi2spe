import json
import csv
import collections
import statistics

import tqdm


if __name__ == '__main__':

    query_paper_count_by_mag_field = collections.defaultdict(int)
    pos_paper_count_by_mag_field = collections.defaultdict(int)
    neg_paper_count_by_mag_field = collections.defaultdict(int)

    num_query_paper_ids_found_mag = 0
    num_pos_paper_ids_found_mag = 0
    num_neg_paper_ids_found_mag = 0

    unique_paper_ids = set()

    extra_metadata = json.load(open('train_extra_metadata.json', 'r'))

    with open('train_triples.csv', 'r') as f_in:
        reader = csv.reader(f_in, delimiter=',')
        for row in tqdm.tqdm(reader):
            query_paper_id = row[0]
            pos_paper_id = row[1]
            neg_paper_id = row[2]

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

    print('num_query_paper_ids_found_mag=', str(num_query_paper_ids_found_mag))
    print('num_pos_paper_ids_found_mag=', str(num_pos_paper_ids_found_mag))
    print('num_neg_paper_ids_found_mag=', str(num_neg_paper_ids_found_mag))
    print()

    print('query_paper_count_by_mag_field')
    print(query_paper_count_by_mag_field)
    print()

    print('pos_paper_count_by_mag_field')
    print(pos_paper_count_by_mag_field)
    print()

    print('neg_paper_count_by_mag_field')
    print(neg_paper_count_by_mag_field)
    print()
