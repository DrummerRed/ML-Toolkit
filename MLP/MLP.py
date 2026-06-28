class MLP:
    def __init__(self):
        self.__layers = 0                                           # Количество слоев нейросети
        self.__neuron_counts = list()                               # Количество нейронов в каждом слое
        self.__activations = list()                                 # Функции активации каждого слоя

    def set_layer(self, neuron_counts, activation_func):
        self.__layers += 1
        self.__neuron_counts.append(neuron_counts)
        self.__activations.append(activation_func)


    def get_settings(self):
        print(f"Layers: {self.__layers}")
        for i in range(self.__layers):
            print(f"{i} layer:\t{self.__neuron_counts[i]} neurons\t Activation func - {self.__activations[i]}")
