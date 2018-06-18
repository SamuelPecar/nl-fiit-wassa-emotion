epochs = 20
batch_size = 64
dim = 200
embeddings_path = 'data/emb.200d.txt'

emoji2word = False

# Partition of train dataset
# partition = 10000
partition = None


# Output formatting

# for silent
# verbose = 0

# for progess bar
verbose = 1

# for one line per epoch
# verbose = 2

# Early stopping
early_stop_monitor = 'val_f1'
early_stop_patience = 2
early_stop_mode = 'max'

# list for convesion of classes
classes = 6
labels_to_index = {
    "sad": 0,
    "joy": 1,
    "disgust": 2,
    "surprise": 3,
    "anger": 4,
    "fear": 5
}

index_to_label = {
    0: "sad",
    1: "joy",
    2: "disgust",
    3: "surprise",
    4: "anger",
    5: "fear"
}
