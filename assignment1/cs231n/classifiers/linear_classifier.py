import numpy as np
from cs231n.classifiers.linear_svm import *
from cs231n.classifiers.softmax import *


class LinearClassifier():
    def __init__(self):
        self.W = None


    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Обучите этот линейный классификатор, используя стохастический
        градиентный спуск.

        Получает на входе:
        - X: numpy массив формой (N, D), содержащий обучающие данные;
          N - количество обучающих точек, размерность которых D.
        - y: numpy массив формой (N,) содержащий обучающие метки; y[i] = c
          означает, что X[i] имеет метку 0 <= c < C для C классов.
        - learning_rate: (float) скорость обучения оптимизации.
        - reg: (float) сила регуляризации.
        - num_iters: (integer) количество шагов, которые необходимо предпринять
          при оптимизации.
        - batch_size: (integer) количество обучающих точек для использования на
          каждом шаге.
        - verbose: (boolean) Если True, печатает прогресс во время оптимизации.

        Возвращает:
        - loss_history: Список, содержащий значение функции потерь на каждой
          итерации обучения.
        """
        num_train, dim = X.shape
        # предположим, что y принимает значения 0...K-1,
        # где K - количество классов
        num_classes = np.max(y) + 1
        if self.W is None:
            # лениво инициализируем W
            self.W = np.random.randn(dim, num_classes) * 0.001

        # Запустите стохастический градиентный спуск для оптимизации W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            # ***************************** ЗАДАНИЕ ***************************** #
            # Выберите batch_size элементов из обучающих данных и соответствующих #
            # им меток, чтобы использовать их в этом раунде градиентного спуска.  #
            # Сохраните данные в X_batch, а соответствующие им метки в y_batch;   #
            # После выборки X_batch должен иметь форму (batch_size, D),           #
            # а y_batch форму (batch_size,).                                      #
            #                                                                     #
            # ПОДСКАЗКА: Используйте np.random.choice(), чтобы генерировать       #
            # индексы. Выборка с перестановкой быстрее, чем без неё.              #

            # ************************ НАЧАЛО МОЕГО КОДА ************************ #
            rng = np.random.default_rng()
            batch_indices = rng.choice(num_train, size=batch_size, replace=False)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            # ************************ КОНЕЦ МОЕГО КОДА ************************* #

            # оценим потерю и градиент
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update
            # ***************************** ЗАДАНИЕ ***************************** #
            # Обновите веса, используя градиент и скорость обучения.              #

            # ************************ НАЧАЛО МОЕГО КОДА ************************ #
            pass
            # ************************ КОНЕЦ МОЕГО КОДА ************************* #

            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))

        return loss_history


    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Получает на входе:
        - X: numpy массив формой (N, D), содержащий обучающие данные;
          N - количество обучающих точек, размерность которых D.

        Возвращает:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        y_pred = np.zeros(X.shape[0])

        # ***************************** ЗАДАНИЕ ***************************** #
        # Реализуйте этот метод. Сохраните предсказанные метки в y_pred.      #

        # ************************ НАЧАЛО МОЕГО КОДА ************************ #
        pass
        # ************************ КОНЕЦ МОЕГО КОДА ************************* #

        return y_pred


    def loss(self, X_batch, y_batch, reg):
        """
        Вычисляет функцию потерь и её производную.
        Подклассы переопределят этот метод.

        Получает на входе:
        - X_batch: numpy массив формой (batch_size, D), содержащий минипакет
          из batch_size точек данных, размерность которых D.
        - y_batch: numpy массив формой (batch_size,), содержащий метки для
          минипакета.
        - reg: (float) сила регуляризации.

        Возвращает кортеж из:
        - loss: (float) потери.
        - grad: градиента по весам W; массив такой же формы как W.
        """
        pass


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """
    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """
    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
