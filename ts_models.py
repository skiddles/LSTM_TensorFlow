from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import os
from abc import ABCMeta
from abc import abstractmethod
from tabulate import tabulate


class MultiVariate:

    class LSTMs:

        class Simple:
            def __init__(self):

                # Originally, input_shape_1 was train_X.shape[1]
                # and input_shape_2 was train_X.shape[2]
                self.random_state = None
                self.model = None
                self.train_X = None
                self.train_y = None
                self.test_X = None
                self.test_y = None
                self.predictions = None
                self.scores = None
                self.history = None

            def define_model(self, input_shape_1, input_shape_2, random_state=61):
                """
                This initializes the Multivariate.LSTM.Simple model

                :param input_shape_1: Originally, train_X.shape[1], this is the length of the second dimension
                                      of the training dataset
                :param input_shape_2: Originally, train_X.shape[2], this is the length of the third dimension
                                      of the training dataset
                :param random_state: A parameter to set the random seed parameter
                """
                self.random_state = random_state
                self.model = Sequential()
                self.model.add(LSTM(50, input_shape=(input_shape_1, input_shape_2)))
                self.model.add(Dense(1))
                self.model.compile(loss='mae', optimizer='adam')

            def set_train_data(self, train_data):
                self.train_X, self.train_y = train_data

                # self.scores = self.model.evaluate(X, Y, verbose=0)
                # print("%s: %.2f%%" % (self.model.metrics_names[1], self.scores[1] * 100))

            def set_test_data(self, test_data):
                self.test_X, self.test_y = test_data

            def train(self, epochs=150, batch_size=50, verbose=0):
                assert self.train_X is not None
                assert self.train_y is not None
                self.history = self.model.fit(self.train_X,
                                              self.train_y,
                                              epochs=epochs,
                                              batch_size=batch_size,
                                              validation_data=(self.test_X, self.test_y),
                                              verbose=verbose,
                                              shuffle=False)

            def predict(self):
                self.predictions = self.model.predict(self.test_X)
                print(self.predictions[-5:, :])

            def save(self, filename):
                filenm, extn = os.path.splitext(filename)
                model_json = self.model.to_json()
                with open(filename, "w") as json_file:
                    json_file.write(model_json)
                # serialize weights to HDF5
                wt_filename = '.'.join([filenm, "h5"])
                print("Saving weights and biases to '%s'" % wt_filename)
                self.model.save_weights(wt_filename)
                print("Saved model to disk")

            def load(self, filename):
                filenm, extn = os.path.splitext(filename)
                json_file = open(filename, 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                self.model = model_from_json(loaded_model_json)
                # load weights into new model
                wt_filename = '.'.join([filenm, "h5"])
                print("Loading weights and biases from '%s'" % wt_filename)
                self.model.load_weights(wt_filename)
                print("Loaded model from disk")

            def evaluate_model(self):
                # evaluate loaded model on test data
                self.model.compile(loss='mae', optimizer='adam')
                self.scores = self.model.evaluate(self.test_X, self.test_y, verbose=2)
                print(self.model.metrics_names)
                # print("%s: %.2f%%" % (self.model.metrics_names[1]))
                # print("%s: %.2f%%" % (self.model.metrics_names[1], self.scores[1] * 100))
