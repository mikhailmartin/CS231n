from random import shuffle
import numpy as np


def svm_loss_naive(W, X, y, reg, delta=1):
    """
    Структурированная функция потерь SVM, наивная реализация (с циклами).

    Входы имеют размерность D, есть C классов, и мы работает с минипакетами
    из N примеров.

    Получает на входе:
    - W: numpy массив формой (D, C), содержащий веса.
    - X: numpy массив формой (N, D), содержащий минипакет данных.
    - y: numpy массив формой (N,), содержащий обучающие метки; y[i] = c
      означает, что X[i] имеет метку c, где 0 <= c < C.
    - reg: (float) сила регуляризации.

    Возвращает кортеж из:
    - loss: (float) потери.
    - grad: градиента по весам W; массив такой же формы как W.
    """
    # Инициализируем потерю и градиент нулями
    loss = 0.0
    grad = np.zeros_like(W)

    # вычислим потерю и градиент
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + delta
            if margin > 0:
                loss += margin

                # Градиент для правильного класса = -n * X[i], где n -
                # количество неправильных классов, для которых перевес > 0.
                # Только здесь подсчёт осуществляется не через умножение, а
                # через сложение.
                grad[:, y[i]] += -X[i]
                # градиент для неправильного класса = X[i], если перевес > 0
                #                                   = 0 иначе
                grad[:, j] += X[i]

    # Прямо сейчас loss - это сумма по всем обучающим примерам, но мы хотим,
    # чтобы она была средней, поэтому мы делим на num_train.
    loss /= num_train
    grad /= num_train  # МОЙ КОД

    # Добавим регуляризацию к потере
    loss += reg * np.sum(W * W)
    grad += reg * W  # МОЙ КОД

    # ***************************** ЗАДАНИЕ ***************************** #
    # Вычислите градиент функции потерь и сохраните его в grad. Вместо    #
    # того, чтобы сначала вычислять потерю, а потом производную, может    #
    # быть проще вычислять производную одновременно с потерей.            #
    # В результате вам, возможно, потребуется изменить приведённый выше   #
    # код, чтобы вычислить градиент.                                      #

    # ************************ НАЧАЛО МОЕГО КОДА ************************ #
    pass
    # ************************ КОНЕЦ МОЕГО КОДА ************************* #

    return loss, grad


def svm_loss_vectorized(W, X, y, reg, delta=1):
    """
    Структурированная функция потерь SVM, векторизированная реализация.

    Получает на входе:
    - W: numpy массив формой (D, C), содержащий веса.
    - X: numpy массив формой (N, D), содержащий минипакет данных.
    - y: numpy массив формой (N,), содержащий обучающие метки; y[i] = c
      означает, что X[i] имеет метку c, где 0 <= c < C.
    - reg: (float) сила регуляризации.

    Возвращает кортеж из:
    - loss: (float) потери.
    - grad: градиента по весам W; массив такой же формы как W.
    """
    # Инициализируем потерю и градиент нулями
    loss = 0.0
    grad = np.zeros_like(W)

    # ***************************** ЗАДАНИЕ ***************************** #
    # Реализуйте векторизованную версию вычисления структурированной      #
    # потери SVM, результат сохраните в loss.                             #
    
    # ************************ НАЧАЛО МОЕГО КОДА ************************ #
    num_train = X.shape[0]

    # получаем оценки перемножением матриц XxW
    scores = X.dot(W)
    # вектор-столбец из оценок правильных классов
    correct_class_scores = scores[np.arange(num_train), y].reshape(num_train, 1)
    # матрица перевесов
    margins = np.maximum(0, scores - correct_class_scores + delta)
    # зануляем перевесы (которые равны delta) для правильных классов
    margins[np.arange(num_train), y] = 0
    # подсчёт средней потери по обучающим точкам
    loss = margins.sum() / num_train
    # добавление штрафа регуляризации
    loss += reg * np.sum(W * W)
    # ************************ КОНЕЦ МОЕГО КОДА ************************* #

    # ***************************** ЗАДАНИЕ ***************************** #
    # Реализуйте векторизованную версию вычисления градиента для          #
    # структурированной потери SVM, сохраняя результат в grad.            #
    #                                                                     #
    # ПОДСКАЗКА: Вместо вычисления градиента с нуля может быть проще      #
    # повторно использовать некоторые промежуточные значения, которые вы  #
    # использовали для вычисления потерь.                                 #
    
    # ************************ НАЧАЛО МОЕГО КОДА ************************ #
    margins_count = np.zeros_like(margins)
    margins_count[margins > 0] = 1
    a = margins_count.sum(axis=1)
    margins_count[np.arange(num_train), y] = -a
    # подсчёт среднего градиента по обучающим точкам
    grad = (X.T).dot(margins_count) / num_train
    # добавляем регуляризацию градиента
    grad += reg * W
    # ************************ КОНЕЦ МОЕГО КОДА ************************* #

    return loss, grad
