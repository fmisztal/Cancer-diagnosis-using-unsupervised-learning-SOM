import numpy as np
import matplotlib.pyplot as plt

from etap2.data_preprocessing import load_data
from etap2.SOM_neural_network import SOM

def test_learning_rate(learning_rate):
    x_train, y_train, x_test, y_test = load_data()
    som = SOM(input_shape=10, output_shape=(10, 10))
    som.train(x_data=x_train, num_epochs=100, learning_rate=learning_rate, x_test=x_test, y_test=y_test)
    return som.accuracy[-1], som.sensitivity[-1], som.specificity[-1]

def test_learning_rate_per_epoch(learning_rate):
    x_train, y_train, x_test, y_test = load_data()
    som = SOM(input_shape=10, output_shape=(10, 10))
    som.train(x_data=x_train, num_epochs=400, learning_rate=learning_rate, x_test=x_test, y_test=y_test)
    return som.accuracy

def test_num_neurons(num_neurons):
    x_train, y_train, x_test, y_test = load_data()
    som = SOM(input_shape=10, output_shape=(num_neurons, num_neurons))
    som.train(x_data=x_train, num_epochs=100, learning_rate=0.05, x_test=x_test, y_test=y_test)
    return som.accuracy[-1], som.sensitivity[-1], som.specificity[-1]


start_lr = 0.01
end_lr = 0.4
learning_rates = np.arange(start_lr, end_lr+0.01, 0.01).round(2)

lr_per_epoch = [0.001, 0.01, 0.1, 0.3]
epoch_range = range(1, 401)

min_neurons = 2
max_neurons = 20
num_neurons_range = range(min_neurons, max_neurons+1)


if __name__ == '__main__':
    # Dokładność sieci w zależności od wartości współczynnika uczenia
    accuracy_lr = []
    sensitivity_lr = []
    specificity_lr = []
    for lr in learning_rates:
        print("Testing learning rate:", lr)
        result = test_learning_rate(lr)
        accuracy_lr.append(result[0])
        sensitivity_lr.append(result[1])
        specificity_lr.append(result[2])

    plt.plot(learning_rates, accuracy_lr, marker='o')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Learning Rate')
    plt.show()

    plt.plot(learning_rates, sensitivity_lr, marker='o')
    plt.xlabel('Learning Rate')
    plt.ylabel('Sensitivity')
    plt.title('Sensitivity vs. Learning Rate')
    plt.show()

    plt.plot(learning_rates, specificity_lr, marker='o')
    plt.xlabel('Learning Rate')
    plt.ylabel('Specificity')
    plt.title('Specificity vs. Learning Rate')
    plt.show()


    # Dokładność sieci dla 4 wsp. uczenia w dziedzinie 400 epok
    accuracies_lr_per_ep = []
    for lr in lr_per_epoch:
        print("Testing learning rate:", lr)
        accuracies_lr_per_ep.append(test_learning_rate_per_epoch(lr))

    plt.figure(figsize=(10, 6))
    for i, lr in enumerate(lr_per_epoch):
        plt.plot(epoch_range, accuracies_lr_per_ep[i], label='Learning Rate: {}'.format(lr), linewidth=0.8)

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Learning Rate per Epoch')
    plt.legend()
    plt.show()


    # Dokładność sieci w zależności od liczby neuronów w sieci
    accuracies_nn = []
    sensitivity_nn = []
    specificity_nn = []
    for num_neurons in num_neurons_range:
        print("Testing number of neurons:", num_neurons)
        results = test_num_neurons(num_neurons)
        accuracies_nn.append(results[0])
        sensitivity_nn.append(results[1])
        specificity_nn.append(results[2])

    plt.plot(num_neurons_range, accuracies_nn, marker='o')
    plt.xlabel('Number of neurons in one grid dimension')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Number of Neurons')
    x_ticks = np.arange(min_neurons, max_neurons + 1, 2)
    plt.xticks(x_ticks)
    plt.show()

    plt.plot(num_neurons_range, sensitivity_nn, marker='o')
    plt.xlabel('Number of neurons in one grid dimension')
    plt.ylabel('Sensitivity')
    plt.title('Sensitivity vs. Number of Neurons')
    x_ticks = np.arange(min_neurons, max_neurons + 1, 2)
    plt.xticks(x_ticks)
    plt.show()

    plt.plot(num_neurons_range, specificity_nn, marker='o')
    plt.xlabel('Number of neurons in one grid dimension')
    plt.ylabel('Specificity')
    plt.title('Specificity vs. Number of Neurons')
    x_ticks = np.arange(min_neurons, max_neurons + 1, 2)
    plt.xticks(x_ticks)
    plt.show()
