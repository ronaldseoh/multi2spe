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

    query_paper_ids_seen = set()
    pos_paper_ids_seen = set()
    neg_paper_ids_seen = set()

    num_query_paper_ids_found_mag = 0
    num_pos_paper_ids_found_mag = 0
    num_neg_paper_ids_found_mag = 0

    num_triples_pos_cross_domain_pure = 0
    num_triples_neg_cross_domain_pure = 0

    num_triples_pos_cross_domain = 0
    num_triples_neg_cross_domain = 0

    unique_paper_ids = set()

    extra_metadata = json.load(open('preprocessed/train_extra_metadata.json', 'r'))

    with open("preprocessed/data-train.p", 'rb') as f_in:
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

                if query_paper_id not in query_paper_ids_seen and query_paper_id in extra_metadata.keys() and extra_metadata[query_paper_id]['mag_field_of_study'] is not None:
                    num_query_paper_ids_found_mag += 1

                    for f in extra_metadata[query_paper_id]['mag_field_of_study']:
                        query_paper_count_by_mag_field[f] += 1

                else:
                    if query_paper_id not in query_paper_ids_seen:
                        query_paper_count_by_mag_field['**Unknown**'] += 1

                if pos_paper_id in extra_metadata.keys() and extra_metadata[pos_paper_id]['mag_field_of_study'] is not None:
                    if pos_paper_id not in pos_paper_ids_seen:
                        num_pos_paper_ids_found_mag += 1

                        for f in extra_metadata[pos_paper_id]['mag_field_of_study']:
                            pos_paper_count_by_mag_field[f] += 1

                    if query_paper_id in extra_metadata.keys() and extra_metadata[query_paper_id]['mag_field_of_study'] is not None:
                        if len(set(extra_metadata[query_paper_id]['mag_field_of_study']).intersection(extra_metadata[pos_paper_id]['mag_field_of_study'])) == 0:
                            num_triples_pos_cross_domain_pure += 1

                        if len(set(extra_metadata[pos_paper_id]['mag_field_of_study']) - set(extra_metadata[query_paper_id]['mag_field_of_study'])) > 0:
                            num_triples_pos_cross_domain += 1

                else:
                    if pos_paper_id not in pos_paper_ids_seen:
                        pos_paper_count_by_mag_field['**Unknown**'] += 1

                if neg_paper_id in extra_metadata.keys() and extra_metadata[neg_paper_id]['mag_field_of_study'] is not None:
                    if neg_paper_id not in neg_paper_ids_seen:
                        num_neg_paper_ids_found_mag += 1

                        for f in extra_metadata[neg_paper_id]['mag_field_of_study']:
                            neg_paper_count_by_mag_field[f] += 1

                    if query_paper_id in extra_metadata.keys() and extra_metadata[query_paper_id]['mag_field_of_study'] is not None:
                        if len(set(extra_metadata[query_paper_id]['mag_field_of_study']).intersection(extra_metadata[neg_paper_id]['mag_field_of_study'])) == 0:
                            num_triples_neg_cross_domain_pure += 1

                        if len(set(extra_metadata[neg_paper_id]['mag_field_of_study']) - set(extra_metadata[query_paper_id]['mag_field_of_study'])) > 0:
                            num_triples_neg_cross_domain += 1

                else:
                    if neg_paper_id not in neg_paper_ids_seen:
                        neg_paper_count_by_mag_field['**Unknown**'] += 1

                query_paper_ids_seen.add(query_paper_id)
                pos_paper_ids_seen.add(pos_paper_id)
                neg_paper_ids_seen.add(neg_paper_id)

            except EOFError:
                break

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

    print('num_triples_pos_cross_domain_pure=', str(num_triples_pos_cross_domain_pure))
    print('num_triples_neg_cross_domain_pure=', str(num_triples_neg_cross_domain_pure))
    print('num_triples_pos_cross_domain=', str(num_triples_pos_cross_domain))
    print('num_triples_neg_cross_domain=', str(num_triples_neg_cross_domain))
    print()
