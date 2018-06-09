import tensorflow as tf


def precision(y_true, y_pred):
    K = tf.keras.backend
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    K = tf.keras.backend
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    K = tf.keras.backend
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * ((prec * rec) / (prec + rec))


def calculate_prf(gold, prediction):
    # initialize counters
    labels = set(gold + prediction)
    print("Labels: " + ';'.join(labels))
    tp = dict.fromkeys(labels, 0.0)
    fp = dict.fromkeys(labels, 0.0)
    fn = dict.fromkeys(labels, 0.0)
    precision = dict.fromkeys(labels, 0.0)
    recall = dict.fromkeys(labels, 0.0)
    f = dict.fromkeys(labels, 0.0)
    # check every element
    for g, p in zip(gold, prediction):
        #        print(g,p)
        # TP
        if (g == p):
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    # print stats
    print("Label\tTP\tFP\tFN\tP\tR\tF")
    for label in labels:
        recall[label] = 0.0 if (tp[label] + fn[label]) == 0.0 else (tp[label]) / (tp[label] + fn[label])
        precision[label] = 1.0 if (tp[label] + fp[label]) == 0.0 else (tp[label]) / (tp[label] + fp[label])
        f[label] = 0.0 if (precision[label] + recall[label]) == 0 else (2 * precision[label] * recall[label]) / (precision[label] + recall[label])
        print(label +
              "\t" + str(int(tp[label])) +
              "\t" + str(int(fp[label])) +
              "\t" + str(int(fn[label])) +
              "\t" + str(round(precision[label], 3)) +
              "\t" + str(round(recall[label], 3)) +
              "\t" + str(round(f[label], 3))
              )
        # micro average
        microrecall = (sum(tp.values())) / (sum(tp.values()) + sum(fn.values()))
        microprecision = (sum(tp.values())) / (sum(tp.values()) + sum(fp.values()))
        microf = 0.0 if (microprecision + microrecall) == 0 else (2 * microprecision * microrecall) / (microprecision + microrecall)
    # Micro average
    print("MicAvg" +
          "\t" + str(int(sum(tp.values()))) +
          "\t" + str(int(sum(fp.values()))) +
          "\t" + str(int(sum(fn.values()))) +
          "\t" + str(round(microprecision, 3)) +
          "\t" + str(round(microrecall, 3)) +
          "\t" + str(round(microf, 3))
          )
    # Macro average
    macrorecall = sum(recall.values()) / len(recall)
    macroprecision = sum(precision.values()) / len(precision)
    macroF = sum(f.values()) / len(f)
    print("MacAvg" +
          "\t" + str() +
          "\t" + str() +
          "\t" + str() +
          "\t" + str(round(macroprecision, 3)) +
          "\t" + str(round(macrorecall, 3)) +
          "\t" + str(round(macroF, 3))
          )
    print("Official result:", macroF)
