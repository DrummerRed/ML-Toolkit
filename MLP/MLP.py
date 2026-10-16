import numpy as np

class MLP:

    # Ограничения на вход: предполагается, что модель работает с numpy массивами (np.array)

    def __init__(self):
        
        self.__layers = 0                                           # Количество слоев нейросети
        self.__neuron_counts = list()                               # Количество нейронов в каждом слое
        self.__activations = list()                                 # Функции активации каждого слоя
        self.__finished_settings = False
        self.__types_activations = ["Identity", "Sigmoid", "ReLU"]
        self.__weights_array = list()

    def set_layer(self, neuron_counts, activation_func):            # Задание слоя модели
        if activation_func in self.__types_activations:
            self.__layers += 1
            self.__neuron_counts.append(neuron_counts)
            self.__activations.append(activation_func)
        else:
            raise Exception("Parameter \"activatoin_func\" is set incorrectly")

    def finish_settings(self):                                      # Окончание настройки модели
        if (self.__layers != 0):
            self.__finished_settings = True
        else:
            raise Exception("Layers are missing! Set up a neural network!")

    def get_settings(self):                                         # Вывод архитектуры модели
        if self.__finished_settings:
            print("Model is set")
        else:
            print("Model is not set")

        print(f"Layers: {self.__layers}")
        for i in range(self.__layers):
            print(f"{i} layer:\t{self.__neuron_counts[i]} neurons\tActivation func - {self.__activations[i]}")

    def weights_initialisation(self, x):                            # Инициализация весов каждого слоя
        data_shape = x.shape
        first_weight = np.random.randn(data_shape[1], self.__neuron_counts[0])
        self.__weights_array.append(first_weight)

        for i in range(1, self.__layers):
            weight = np.random.rand(self.__neuron_counts[i-1], self.__neuron_counts[i])
            self.__weights_array.append(weight)

    def get_weights(self, layer_number):                        ######## слои должны быть с 1 до n
        if (-1 < layer_number) and (layer_number < self.__layers):
            return self.__weights_array[layer_number]
        else:
            raise Exception("Parameter \"layer_number\" is incorrect!")

    # def forward(self, x):
    #     if self.__finished_settings:


    # Activation functions
    def identity(self, x):
        return x

    def ReLU(self, x):
        mask = x <= 0
        x[mask] = 0
        return x

    def sigmoid(self, x):
        from math import e
        result = 1 / (1 + e ** (-x))
        return result