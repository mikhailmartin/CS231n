import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Функция потерь Softmax, наивная реализация (с циклами).

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
    - grad: градиента по весам W; массив такой же формы как W.
    """
    # Инициализируем потерю и градиент нулями.
    loss = 0.0
    grad = np.zeros_like(W)
    # **************************** ЗАДАНИЕ **************************** #
    # Вычислите softmax потерю и её градиент, используя явные циклы.    #
    # Сохраните потери в loss, а градиент в dW. Если вы будете здесь    #
    # неосторожны, вы можете легко столкнуться с проблемой числовой     #
    # стабильности. Не забудьте про регуляризацию!                      #

    # *********************** НАЧАЛО МОЕГО КОДА *********************** #
    pass
    # *********************** КОНЕЦ МОЕГО КОДА ************************ #

    return loss, grad


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
