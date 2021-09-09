import pickle

unpickler = pickle.Unpickler(open('train.pkl', 'rb'))

instance_count = 0

instances_fetched = []

while instance_count < 5:
    instance = unpickler.load()
    instances_fetched.append(instance)
    instance_count += 1

for ins in instances_fetched:
    print(ins['source_title'])
