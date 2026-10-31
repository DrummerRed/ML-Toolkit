import numpy as np
import copy

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
        self.__loss_fn = None
        self.__types_loss = ["MSE", "CrossEntropy"]                     # Доступные функции потерь
        self.__dict_of_loss = {
            "MSE": [self.MSE, self.__MSE_derivatiev],
            "CrossEntropy": [self.Cross_entropy, self.Cross_entropy_derivatiev]
        }
        self.__representations = list()                                 # Сохраненные представления модели (без начальной обучающей выборки X)
        self.train_mode = True                                          # Режим обучения/теста модели
        self.__lr = 0.01                                                # Скорость обучения

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
            weight = np.random.randn(self.__neuron_counts[i-1], self.__neuron_counts[i])
            self.__weights_array.append(weight)

    def get_weights(self, layer_number):                        # Вывод значений весов для некторого (layer_number) слоя
        if (-1 < layer_number) and (layer_number < self.__layers):
            return self.__weights_array[layer_number]
        else:
            raise Exception("Parameter \"layer_number\" is incorrect!")

    def forward(self, x):                                               # Прямой проход (возвращает предсказание модели)
        self.__representations.clear()
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
        result = np.copy(x)
        mask = result <= 0
        result[mask] = 0
        return result

    def sigmoid(self, x):
        from math import e
        result = 1 / (1 + e ** (-x))
        return result

    def Softmax(self, x):
        z = np.asarray(x, dtype=float)                   
        z = z - z.max(axis=1, keepdims=True)             
        exp_z = np.exp(z)                                
        return exp_z / exp_z.sum(axis=1, keepdims=True)  

    # Производные функций активации
    def __identity_derivatiev(self, x):
        return np.ones_like(x)

    def __sigmoid_derivatiev(self, x):
        # from math import e
        # result = (e ** x) / (e ** (2 * x) + 2 * (e ** x) + 1)
        result = np.exp(x) / (np.exp(2 * x) + 2 * np.exp(x) + 1)
        return result

    def __ReLU_derivatiev(self, x):
        result = np.copy(x)
        negative_mask =  result <= 0
        positive_mask = result > 0
        result[negative_mask] = 0
        result[positive_mask] = 1
        return result

    # Функции потерь и их градиенты
    def MSE(self, y_true, y_pred):                                  # MSE 
        n = len(y_true)
        temp = (y_true - y_pred) ** 2
        return sum(temp) / n

    def __MSE_derivatiev(self, y_true, y_pred):
        n = len(y_true)
        result = -2 / n * (y_true - y_pred)
        return result

    def Cross_entropy(self, y_true, logits):
        logits = self.Softmax(logits)
        samples_count = logits.shape[0]
        # classes_count = logits.shape[1]
        Loss = 0
        for i in range(samples_count):                              # По идее можно оптимизировать, не используя цикл
            true_class = int(y_true[i][0])
            Loss += np.log(logits[i][true_class] + 1e-12)           # 1e-12, чтоб недопустить log(0)
        return -1 * (Loss / samples_count)

    def Cross_entropy_derivatiev(self, y_true, logits):
        result = None
        if (self.__activations[self.__layers - 1] == "Identity"):
            logits = self.Softmax(logits)
            samples_count = logits.shape[0]
            # classes_count = logits.shape[1]
            one_hot_y_true = np.zeros_like(logits)
            for i in range(samples_count):                          # По идее можно оптимизировать, не используя цикл
                true_class = int(y_true[i][0])
                one_hot_y_true[i][true_class] = 1

            result = (logits -  one_hot_y_true) / samples_count

        else:
           raise Exception("For using CrossEntropy loss the last activation function must be \"Identity\"!") 
        
        return result

    # def test(self, y_true, y_pred):
    #     representation = self.__representations[len(self.__representations)-2]     # вход последнего слоя (он же выход предпоследнего)
    #     return self.__MSE_derivatiev(y_true, y_pred)

    def set_loss_function(self, loss, lr=0.01):
        if loss in self.__types_loss:
            self.__loss_fn = loss
        else:
            print("Parameter loss is incorrect!")

        if isinstance(lr, float) or isinstance(lr, int):
            self.__lr = lr
        else:
            print("Parameter lr is incorrect!")
        return self.__loss_fn

    def backward(self, x, y_true):              # Не забыть обнулять сохраненные представления
        if self.__loss_fn is None:
            print("Loss function is not set! Set up loss fuction!")
        else:
            Weights_grad_array = list()
            y_pred = self.__representations[len(self.__representations)-1]
            Loss = self.__loss_fn
            d_Loss = self.__dict_of_loss[Loss][1]                       # Получаем производную функции потерь
            representation = None
            if self.__layers > 1:  
                Grad = d_Loss(y_true, y_pred)                  # расчет градиента для функции потерь
                for i in reversed(range(self.__layers)):
                    if i > 0:
                        representation = self.__representations[i-1]
                    else:
                        representation = x

                    activation = self.__activations[i]
                    derivatiev_of_activation = self.__dict_activations[activation][1]               # Получаем производную функции активации
                    input_in_layer = np.dot(representation, self.__weights_array[i])
                    Activate_fn_grad = Grad * derivatiev_of_activation(input_in_layer)
                    Weights_grad = np.dot(representation.T, Activate_fn_grad)                           # Расчет градиента весов слоя
                    Weights_grad_array.append(Weights_grad)
                    Grad = np.dot(Activate_fn_grad, self.__weights_array[i].T)
                    # print(Weights_grad)
                    # print("------------")
                    
            Weights_grad_array.reverse()
            self.__update_grads(Weights_grad_array)    

    def __update_grads(self, Weights_grad_array):                              
        length = len(self.__weights_array)
        for i in range(length):
            new_weight = self.__weights_array[i] - self.__lr * Weights_grad_array[i]
            self.__weights_array[i] = new_weight


    # Можно написать вывод изменения loss по итерациям