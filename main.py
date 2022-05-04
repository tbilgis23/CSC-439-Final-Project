from baseline import TextToFeatures, TextToLabels, Classifier
import fancy_model
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score


dataset, info = tfds.load('imdb_reviews', with_info=True,
                          as_supervised=True) # the dataset is also available within tensorflow.
train_dataset, test_dataset_s = dataset['train'], dataset['test']
test_dataset, validation_set = test_dataset_s.take(12500), test_dataset_s.skip(12500)
train_dataset_, test_dataset_ = train_dataset, test_dataset

def baseline_scores(): 
    '''
    This function evaluate the baseline model.
    It will return a dictionary with the scores of the baseline model for bootstrap resampling.
    It will output the f1 score and accuracy of the baseline model.
    @return scores: a dictionary with the scores of the baseline model for bootstrap resampling.
    '''
    train_texts = []
    for i in list(train_dataset_.as_numpy_iterator()):
        train_texts.append([i[0], i[1]])


    train = pd.DataFrame(train_texts, columns=["text", "label"])

    test_texts = []
    for i in list(test_dataset_.as_numpy_iterator()):
        test_texts.append([i[0], i[1]])
      
    test = pd.DataFrame(test_texts, columns=["text", "label"]).groupby("label").sample(frac=0.5) # to get the %25 as testing data

    X_train_, X_test_, Y_train_, Y_test_ = train["text"].tolist(), test["text"].tolist(), train["label"].tolist(), test["label"].tolist()
    train_labels, train_texts = Y_train_, X_train_

    devel_labels, devel_texts = Y_test_, X_test_

    # create the feature extractor and label encoder
    to_features = TextToFeatures(train_texts)
    to_labels = TextToLabels(train_labels)
    # train the classifier on the training data
    classifier = Classifier()
    # print(to_labels(train_labels))
    # print(to_labels(train_texts))
    classifier.train(to_features(train_texts), to_labels(train_labels))

    # make predictions on the development data
    predicted_indices = classifier.predict(to_features(devel_texts))

    # measure performance of predictions
    devel_indices = to_labels(devel_labels)
    spam_label = to_labels.index(0)
    f1 = f1_score(devel_indices, predicted_indices, pos_label=spam_label)
    accuracy = accuracy_score(devel_indices, predicted_indices)

    msg = "\n{:.3%} F1 and {:.3%} accuracy"
    print(msg.format(f1, accuracy))
    score = {}
    j = 0
    for i in predicted_indices:
      if i < 0.5:
        i = 0
      else:
        i = 1
      if i != devel_indices[j]:
        score[devel_texts[j]] = {"baselineScore": 0.0}
      else:
        score[devel_texts[j]] = {"baselineScore" : 1.0}
      j += 1
    return score

BUFFER_SIZE = 10000
BATCH_SIZE = 64

def main():
    '''
    Main function to run the baseline model and experimental model.
    '''
    scores = baseline_scores()

    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    validation_set = validation_set.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    RNN_Model = fancy_model.RNN_Model(train_dataset, test_dataset, validation_set)
    RNN_Model.train(test_dataset, 10)
    RNN_Model.evaluate(test_dataset)
    RNN_Model.predict(test_dataset)
    binary_predictions = RNN_Model.get_binary_predictions()
    
    texts = []
    labels = []

    for z in list(test_dataset_.as_numpy_iterator()):
        texts.append(z[0])
        labels.append(z[1])

    j = 0
    for i in binary_predictions:
        if i != labels[j]:
            if texts[j] in scores:
                scores[texts[j]]["experimentalScore"] = 0.0
        else:
            if texts[j] in scores:
                scores[texts[j]]["experimentalScore"] = 1.0
        j += 1

main()
    

            