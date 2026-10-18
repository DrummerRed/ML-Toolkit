import numpy as np

class MLP:

    # Ограничения на вход: предполагается, что модель работает с numpy массивами (np.array)

    def __init__(self):
        self.__layers = 0                                               # Количество слоев нейросети
        self.__neuron_counts = list()                                   # Количество нейронов в каждом слое
        self.__activations = list()                                     # Функции активации каждого слоя
        self.__finished_settings = False                                # Флаг окончания настройки модели
        self.__types_activations = ["Identity", "Sigmoid", "ReLU"]      # Доступные активации
        self.__dict_activations = {
            "Identity": [self.identity, self.__identity_derivatiev],
            "Sigmoid": [self.sigmoid, self.__sigmoid_derivatiev],
            "ReLU": [self.ReLU, self.__ReLU_derivatiev]
        }
        self.__weights_array = list()                                   # Веса модели
        self.__dict_of_loss = {
            "MSE": [self.MSE]
        }
        self.__representations = list()                                 # Сохраненные представления модели
        self.train_mode = True                                          # Режим обучения/теста модели

    def set_layer(self, neuron_counts, activation_func):                # Задание слоя модели
        if activation_func in self.__types_activations:
            self.__layers += 1
            self.__neuron_counts.append(neuron_counts)
            self.__activations.append(activation_func)
        else:
            raise Exception("Parameter \"activatoin_func\" is set incorrectly")

    def finish_settings(self, x):                                          # Окончание настройки модели
        if (self.__layers != 0):                                           # Для завершения настройки необходимо передать полный набор или батч данных для обучения
            self.__weights_initialisation(x)
            self.__finished_settings = True
        else:
            raise Exception("Layers are missing! Set up a neural network!")

    def get_settings(self):                                             # Вывод архитектуры модели
        if self.__finished_settings:
            print("Model is set")
        else:
            print("Model is not set!")

        print(f"Layers: {self.__layers}")
        for i in range(self.__layers):
            print(f"{i} layer:\t{self.__neuron_counts[i]} neurons\tActivation func - {self.__activations[i]}")

    def __weights_initialisation(self, x):                                # Инициализация весов каждого слоя
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

    def forward(self, x):                                               # Прямой проход (возвращает предсказание модели)
        if self.__finished_settings:
            result = x
            for i in range(self.__layers):
                w = self.__weights_array[i]
                activate_func = self.__dict_activations[self.__activations[i]][0]
                result = self.__dot_product(result, w, activate_func)
                if self.train_mode == True:
                    self.__representations.append(result)
        else:
            result = None
            raise Exception("Model is not set! Set up a neural network!")
        return result

    def __dot_product(self, x, w, activate_func):                       # Перемноежение матрицы признаков на матрицу весов и применение функции активации
        return activate_func(np.dot(x, w))

    # Функции активации
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

    # Производные функций активации
    def __identity_derivatiev(self, x):
        return np.ones_like(x)

    def __sigmoid_derivatiev(self, x):
        from math import e
        result = (e ** x) / (e ** (2 * x) + 2 * (e ** x) + 1)
        return result

    def __ReLU_derivatiev(self, x):
        negative_mask =  x <= 0
        positive_mask = x > 0
        x[negative_mask] = 0
        x[positive_mask] = 1
        return x

    # Функции потерь и их градиенты
    def MSE(self, y_true, y_pred):                                  # MSE 
        n = len(y_true)
        temp = (y_true - y_pred) ** 2
        return sum(temp) / n

    def __MSE_derivatiev(self, representation, y_true, y_pred):
        x = representation
        n = len(y_true)
        result = -2 / n * np.dot(x.T, (y_true - y_pred))
        # print(f"x.T: {x.T}")
        # print(f"y_true - y_pred:\n{y_true - y_pred}")
        # print("Градиент:")
        return result

    def test(self, y_true, y_pred):
        representation = self.__representations[len(self.__representations)-2]     # вход последнего слоя
        # print(representation)
        # print(self.__MSE_derivatiev(representation, y_true, y_pred))
        return self.__MSE_derivatiev(representation, y_true, y_pred)