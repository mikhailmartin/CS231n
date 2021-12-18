import abc

import numpy as np
from cs231n.classifiers.svm import svm_loss_vectorized
from cs231n.classifiers.softmax import softmax_loss_vectorized


class LinearClassifier(metaclass=abc.ABCMeta):
    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """Обучает этот линейный классификатор, используя стохастический градиентный спуск.

        Args:
            X: numpy массив формой (N, D), содержащий обучающие данные;
              N - количество обучающих точек, размерность которых D.
            y: numpy массив формой (N,) содержащий обучающие метки; y[i] = c означает, что X[i]
              имеет метку 0 <= c < C для C классов.
            learning_rate: (float) скорость обучения оптимизации.
            reg: (float) сила регуляризации.
            num_iters: (integer) количество шагов, которые необходимо предпринять при оптимизации.
            batch_size: (integer) количество обучающих точек для использования на каждом шаге.
            verbose: (boolean) Если True, печатает прогресс во время оптимизации.

        Returns:
            loss_history: Список, содержащий значение функции потерь на каждой итерации обучения.
        """
        rng = np.random.default_rng()
        num_train, dim = X.shape
        # предположим, что y принимает значения 0...K-1,
        # где K - количество классов
        num_classes = np.max(y) + 1
        if self.W is None:
            # лениво инициализируем W
            self.W = rng.standard_normal(size=(dim, num_classes)) * 0.001

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
            batch_indices = rng.choice(num_train, size=batch_size, replace=False)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            # ************************ КОНЕЦ МОЕГО КОДА ************************* #

            # оценим потерю и градиент
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # Выполните обновление параметров
            # ***************************** ЗАДАНИЕ ***************************** #
            # Обновите веса, используя градиент и скорость обучения.              #

            # ************************ НАЧАЛО МОЕГО КОДА ************************ #
            self.W = self.W - learning_rate * grad
            # ************************ КОНЕЦ МОЕГО КОДА ************************* #

            if verbose and it % 100 == 0:
                print(f'итерация {it}/{num_iters}: потеря {loss}')

        return loss_history

    def predict(self, X):
        """Использует обученные веса этого линейного классификатора для прогнозирования меток для
        точек данных.

        Args:
            X: numpy массив формой (N, D), содержащий обучающие данные;
              N - количество обучающих точек, размерность которых D.

        Returns:
            y_pred: Предсказанные метки для точек данных в X. y_pred - это одномерный массив
              длиной N, а каждый элемент является целым числом, которое кодирует предсказанный
              класс.
        """
        y_pred = np.zeros(X.shape[0])

        # ***************************** ЗАДАНИЕ ***************************** #
        # Реализуйте этот метод. Сохраните предсказанные метки в y_pred.      #

        # ************************ НАЧАЛО МОЕГО КОДА ************************ #
        scores = np.dot(X, self.W)
        y_pred = scores.argmax(axis=1)
        # ************************ КОНЕЦ МОЕГО КОДА ************************* #

        return y_pred

    @abc.abstractmethod
    def loss(self, X_batch, y_batch, reg):
        """Вычисляет функцию потерь и её производную. Подклассы переопределят этот метод.

        Args:
            X_batch: numpy массив формой (batch_size, D), содержащий минипакет из batch_size точек
              данных, размерность которых D.
            y_batch: numpy массив формой (batch_size,), содержащий метки для минипакета.
            reg: (float) сила регуляризации.

        Returns:
            loss: (float) потери.
            grad: градиента по весам W; массив такой же формы как W.
        """
        raise NotImplementedError()


class SVM(LinearClassifier):
    """ Подкласс, использующий функцию потерь мультиклассовой SVM """
    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """ Подкласс, использующий функцию потерь Softmax + Кросс-энтропийных """
    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
