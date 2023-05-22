import numpy as np
import matplotlib.pyplot as plt

from data_preprocessing import load_data

class SOM:
    def __init__(self, input_shape, output_shape):
        self.weights = np.random.rand(output_shape[0], output_shape[1], input_shape)
        self.output_shape = output_shape
        self.accuracy = []
        self.sensitivity = []
        self.specificity = []
        self.mean_squared_error = []

    def train(self, x_data, num_epochs, learning_rate, x_test, y_test):
        for epoch in range(num_epochs):
            self._predict(x_test, y_test)
            best_matching_unit = self._find_best_matching_unit(x_data[epoch])
            self._update_weights(x_data[epoch], best_matching_unit, learning_rate)

        self._show_results()

    def _find_best_matching_unit(self, sample):
        min_dist = np.inf
        best_unit = (0, 0)
        for i in range(self.output_shape[0]):
            for j in range(self.output_shape[1]):
                dist = np.linalg.norm(sample - self.weights[i, j])
                if dist < min_dist:
                    min_dist = dist
                    best_unit = (i, j)
        return best_unit

    def _update_weights(self, sample, best_unit, learning_rate):
        for i in range(self.output_shape[0]):
            for j in range(self.output_shape[1]):
                dist = np.linalg.norm(np.array(best_unit) - np.array([i, j]))
                influence = self._calculate_influence(dist)
                self.weights[i, j] += learning_rate * influence * (sample - self.weights[i, j])

    def _calculate_influence(self, dist):
        sigma = np.mean(self.output_shape) / 2
        return np.exp(-dist ** 2 / (2 * sigma ** 2))

    def _predict(self, x_data, y_data):
        predictions = []
        for sample, label in zip(x_data, y_data):
            best_matching_unit = self._find_best_matching_unit(sample)
            if np.mean(self.weights[best_matching_unit]) > 0.3:
                predictions.append(1)
            else:
                predictions.append(0)

        self._calculate_indicators(predictions, y_data)

    def _calculate_indicators(self, predictions, y_data):
        TP = FP = TN = FN = 0
        for prediction, y in zip(predictions, y_data):
            if prediction == 1 and y == 1:
                TP += 1
            elif prediction == 1 and y == 0:
                FP += 1
            elif prediction == 0 and y == 1:
                FN += 1
            elif prediction == 0 and y == 0:
                TN += 1

        # Czułość
        sensitivity = TP / (TP + FN)
        self.sensitivity.append(sensitivity * 100)

        # Specyficzność
        specificity = TN / (TN + FP)
        self.specificity.append(specificity * 100)

        # Dokładność
        accuracy = np.mean(predictions == y_data)
        self.accuracy.append(accuracy * 100)

        #Błąd średnio kwadratowy
        mean_squared_error = np.mean((y_data - predictions) ** 2)
        self.mean_squared_error.append(mean_squared_error * 100)



    def _show_results(self):
        self._visualize_som()
        self._show_indicators()
        print("Classification accuracy: {:.2f}%".format(self.accuracy[-1]))
        print("Classification sensitivity: {:.2f}%".format(self.sensitivity[-1]))
        print("Classification specificity: {:.2f}%".format(self.specificity[-1]))
        print("Classification mean squared error: {:.2f}".format(self.mean_squared_error[-1]))

    def _visualize_som(self):
        heatmap = np.zeros(self.output_shape)
        for i in range(self.output_shape[0]):
            for j in range(self.output_shape[1]):
                heatmap[i, j] = np.mean(self.weights[i, j])  # Średnia wartość wag dla danego neuronu
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title('Heatmap of neuron weights')
        plt.show()

    def _show_indicators(self):
        plt.plot(range(len(self.accuracy)), self.accuracy)
        plt.plot(range(len(self.mean_squared_error)), self.mean_squared_error)
        plt.title('Accuracy')
        plt.xlabel('Number of epochs')
        plt.ylabel('Accuracy [%]')
        plt.show()

        plt.plot(range(len(self.specificity)), self.specificity, label='Specificity')
        plt.plot(range(len(self.sensitivity)), self.sensitivity, label='Sensitivity')
        plt.legend()
        plt.title('Sensitivity and specificity')
        plt.xlabel('Number of epochs')
        plt.ylabel('Indicator [%]')
        plt.show()

        plt.plot(range(len(self.mean_squared_error)), self.mean_squared_error)
        plt.title('Mean squared error')
        plt.xlabel('Number of epochs')
        plt.ylabel('Mean squared error')
        plt.show()


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()
    som = SOM(input_shape=10, output_shape=(10, 10))
    som.train(x_data=x_train, num_epochs=200, learning_rate=0.05, x_test=x_test, y_test=y_test)
