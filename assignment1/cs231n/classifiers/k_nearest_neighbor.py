import numpy as np


class KNearestNeighbor():
    """ kNN-классификатор с L2-расстоянием """
    def __init__(self):
        pass


    def train(self, X, y):
        """
        Обучает классификатор. Для k-NN это просто запоминание обучающих данных.

        Получает на входе:
        - X: numpy массив формой (num_train, D), содержащий обучающие данные,
          состоящие из num_train примеров каждого измерения D.
        - y: numpy массив формой (N,), содержащий обучающие метки, где
          y[i] - метка для X[i].
        """
        self.X_train = X
        self.y_train = y


    def predict(self, X, k=1, num_loops=0):
        """
        Предсказывает метки для тестовых данных с помощью этого классификатора.

        Получает на входе:
        - X: numpy массив формой (num_test, D), содержащий тестовые данные,
          состоящие из num_test примеров каждого измерения D.
        - k: Количество ближайших соседей, голосующих за предсказанные метки.
        - num_loops: Определяет, какую реализацию использовать для вычисления
          расстояний между обучающими и тестовыми точками.

        Возвращает:
        - y: numpy массив формой (num_test,), содержащий предсказанные метки для
          тестовых данных, где y[i] - предсказанная метка для тестовой точки X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)


    def compute_distances_two_loops(self, X):
        """
        Вычисляет расстояние между каждой тестовой точкой в X и каждой обучающей
        точкой в self.X_train, используя вложенный цикл как по обучающим, так и по
        тестовым данным.

        Получает на входе:
        - X: numpy массив формой (num_test, D), содержащий тестовые данные.

        Возвращает:
        - dists: numpy массив формой (num_test, num_train), в котором dists[i, j] -
          евклидово расстояние между i-ой тестовой и j-ой обучающей точкой.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                ################################################################
                # TODO:                                                        #
                # Вычислите L2-расстояние между i-ой тестовой и j-ой обучающей #
                # точкой, и сохраните результат в dists[i, j]. Вы не должны    #
                # использовать цикл по размерности точки данных или функцию    #
                # np.linalg.norm().                                            #
                ################################################################
                # ******************** НАЧАЛО МОЕГО КОДА ********************* #
                # формула L2-расстояния
                dist = pow(pow((X[i] - self.X_train[j]), 2).sum(), 0.5)
                dists[i, j] = dist
                # ******************** КОНЕЦ  МОЕГО КОДА ********************* #

        return dists


    def compute_distances_one_loop(self, X):
        """
        Вычисляет расстояние между каждой тестовой точкой в X и каждой обучающей
        точкой в self.X_train, используя один цикл по тестовым данным.

        Получает на входе:
        - X: numpy массив формой (num_test, D), содержащий тестовые данные.

        Возвращает:
        - dists: numpy массив формой (num_test, num_train), в котором dists[i, j] -
          евклидово расстояние между i-ой тестовой и j-ой обучающей точкой.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            ##############################################################
            # TODO:                                                      #
            # Вычислите L2-расстояние между i-ой тестовой точкой и всеми #
            # обучающими точками, и сохраните результат в dists[i, :].   #
            # Нельзя использовать np.linalg.norm().                      #
            ##############################################################
            # ******************* НАЧАЛО МОЕГО КОДА ******************** #
            pass
            # ******************* КОНЕЦ  МОЕГО КОДА ******************** #
        return dists


    def compute_distances_no_loops(self, X):
        """
        Вычислияет расстояние между каждой тестовой точкой в X и каждой обучающей
        точкой в self.X_train без явных циклов.

        Получает на входе:
        - X: numpy массив формой (num_test, D), содержащий тестовые данные.

        Возвращает:
        - dists: numpy массив формой (num_test, num_train), в котором dists[i, j] -
          евклидово расстояние между i-ой тестовой и j-ой обучающей точкой.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        ##########################################################################
        # TODO:                                                                  #
        # Вычислите L2-расстояние между i-ой тестовой точкой и всеми             #
        # обучающими точками, не используя любуе явные циклы, и сохраните        #
        # результат в dists.                                                     #
        #                                                                        #
        # Вы должны реализовать эту функцию, используя только базовые операции с #
        # массивами; в частности, вы не должны использовать функции из scipy или #
        # np.linalg.norm ().                                                     #
        #                                                                        #
        # ПОДСКАЗКА: Попробуйте сформулировать L2-расстояние, используя          #
        # умножение матриц и две широковещательные (broadcast) суммы.            #
        ##########################################################################

        pass

        return dists


    def predict_labels(self, dists, k=1):
        """
        Учитывая матрицу расстояний между тестовыми и обучающими точками,
        предскажите метку для каждой тестовой точки.

        Получает на входе:
        - dists: numpy массив формой (num_test, num_train), где dists[i, j] -
          даёт расстояние между i-й тестовой и j-й обучающей точкой.

        Возвращает:
        - y: numpy массив формой (num_test,), содержащий предсказанные метки для
          тестовых данных, где y[i] - предсказанная метка для тестовой точки X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # Список длины k, в котором хранятся метки k ближайших соседей для
            # i-й тестовой точки.
            closest_y = []
            #####################################################################
            # TODO:                                                             #
            # Используйте матрицу расстояний, чтобы найти k ближайших соседей   #  
            # i-й тестовой точки, и используйте self.y_train, чтобы найти метки #
            # этих соседей. Храните эти метки в closest_y.                      #
            # ПОДСКАЗКА: Найдите функцию numpy.argsort.                         #
            #####################################################################
            # *********************** НАЧАЛО МОЕГО КОДА *********************** #
            sorted_indices = dists[i].argsort()
            k_closest_indices = sorted_indices[:k]
            closest_y = self.y_train[k_closest_indices]
            # *********************** КОНЕЦ  МОЕГО КОДА *********************** #

            #####################################################################
            # TODO:                                                             #
            # Теперь, когда вы нашли метки k ближайших соседей, вам нужно найти #
            # наиболее распространённую метку в списке меток closest_y.         #
            # Сохраните эту метку в y_pred [i]. При неопределённости выберете   #
            # метку меньшего размера.                                           #
            #####################################################################
            # *********************** НАЧАЛО МОЕГО КОДА *********************** #
            y_pred[i] = np.bincount(closest_y, minlength=10).argmax()
            # *********************** КОНЕЦ  МОЕГО КОДА *********************** #

        return y_pred
