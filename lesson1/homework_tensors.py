
import torch
# Домашнее задание к уроку 1: Основы PyTorch

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

print(torch.arange(16))