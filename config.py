epochs = 20
batch_size = 64
dim = 1024
units = 1024
# embeddings_path = "data/emb.200d.txt"
embeddings_path = "https://tfhub.dev/google/elmo/2"

# Output formatting
verbose = 2

# Early stopping
early_stop_monitor = 'val_acc'
early_stop_patience = 1
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

class_weight = {
    "sad": 1.0,
    "joy": 1.0,
    "disgust": 1.0,
    "surprise": 1.0,
    "anger": 1.0,
    "fear": 1.0
}