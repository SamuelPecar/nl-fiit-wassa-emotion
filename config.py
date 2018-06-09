classes = 6

labels_to_index = {
    "sad": 0,
    "joy": 1,
    "disgust": 2,
    "surprise": 3,
    "anger": 4,
    "fear": 5
}

epochs = 20
batch_size = 64
dim = 200
embeddings_path = 'data/glove.twitter.27B.200d.txt'

partition = 10000
# partition = None
