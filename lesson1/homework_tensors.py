
import torch
# Домашнее задание к уроку 1: Основы PyTorch

# Задание 1: Создание и манипуляции с тензорами (25 баллов)
# 1.1 Создание тензоров (7 баллов)

# Тензор размеро 3x4, заполненный случайными числами от 0 до 1
rand_tensor = torch.rand(3, 4)

# Тензор размером 2x3x4, заполненный нулями
zeros_tensor = torch.zeros(2, 3, 4)

# Тензор размером 5x5, заполненный единицами
ones_tensor = torch.ones(5, 5)

# Тензор размером 4x4 с числами от 0 до 15 (используйте reshape)
reshape_tensor = torch.reshape(torch.arange(16), (4, 4))


# 1.2 Операции с тензорами (6 баллов)
A_tensor = torch.rand(3, 4)
B_tensor = torch.rand(4, 3)

# Транспонирование тензора A
A_tensor_T = A_tensor.T

# Матричное умножение A и B
ABMatMult = A_tensor @ B_tensor

# Поэлементное умножение A и транспонированного B
ABMult = A_tensor * B_tensor.T

# Вычислите сумму всех элементов тензора A
A_tensor_sum = A_tensor.sum()


# 1.3 Индексация и срезы (6 баллов)
tensor555 = torch.rand(5, 5, 5)

# Извлеките первую строку
firstRow = tensor555[0:, 0, 0:]

# Извлеките последний столбец
lastCol = tensor555[0:, 0:, -1:]

# Извлеките подматрицу размером 2x2 из центра тензора
matrix2x2OfTensor555 = tensor555[0:, 1:3, 1:3]

# Извлеките все элементы с четными индексами
evenTensor = tensor555[0:, 0:, ::2]


# 1.4 Работа с формами (6 баллов)
tensor1x24 = torch.rand(24)

# Преобразуйте его в формы:
# 2x12
tensor2x12 = tensor1x24.reshape(2, 12)

# 3x8
tensor3x8 = tensor1x24.reshape(3, 8)

# 4x6
tensor4x6 = tensor1x24.reshape(4, 6)

# 2x3x4
tensor2x3x4 = tensor1x24.reshape(2, 3, 4)

# 2x2x2x3
tensor2x2x2x3 = tensor1x24.reshape(2, 2, 2, 3)
