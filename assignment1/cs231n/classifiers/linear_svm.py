from random import shuffle
import numpy as np


DELTA = 1


def svm_loss_naive(W, X, y, reg):
    """
    Структурированная функция потерь SVM, наивная реализация (с циклами).

    Входы имеют размерность D, есть C классов, и мы работает с минипакетами
    из N примеров.

    Получает на входе:
    - W: numpy массив формой (D, C), содержащий веса.
    - X: numpy массив формой (N, D), содержащий минипакет данных.
    - y: numpy массив формой (N,), содержащий обучающие метки; y[i] = c означает,
      что X[i] имеет метку c, где 0 <= c < C.
    - reg: (float) сила регуляризации.

    Возвращает кортеж из:
    - loss: (float) потери.
    - dW: градиента по весам W; массив такой же формы как W.
    """
    # Инициализируем потерю и градиент нулями.
    loss = 0.0
    dW = np.zeros_like(W)

    # вычислим потерю и градиент
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + DELTA  # DELTA = 1
            if margin > 0:
                loss += margin

    # Прямо сейчас loss - это сумма по всем обучающим примерам, но мы хотим,
    # чтобы она была средней, поэтому мы делим на num_train.
    loss /= num_train

    # Добавим регуляризацию к потере.
    loss += reg * np.sum(W * W)

    # ***************************** ЗАДАНИЕ ***************************** #
    # Вычислите градиент функции потерь и сохраните его в dW. Вместо      #
    # того, чтобы сначала вычислять потерю, а потом производную, может    #
    # быть проще вычислять производную одновременно с потерей.            #
    # В результате вам, возможно, потребуется изменить приведённый выше   #
    # код, чтобы вычислить градиент.                                      #

    # ************************ НАЧАЛО МОЕГО КОДА ************************ #
    pass
    # ************************ КОНЕЦ МОЕГО КОДА ************************* #

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
