import pickle


with open('data-train.p', 'rb') as train_file:
    unpickler = pickle.Unpickler(train_file)
    while True:
        try:
            instance = unpickler.load()
            train_count += 1
        except EOFError:
            break
