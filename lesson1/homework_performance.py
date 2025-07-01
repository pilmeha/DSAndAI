# import time
# import torch
# # Задание 3: Сравнение производительности CPU vs CUDA (20 баллов)

# # 3.1 Подготовка данных (5 баллов)
# bigAMatrix = torch.rand(64, 1024, 1024)
# bigAMatrixAlt = torch.rand(64, 1024, 1024)
# bigBMatrix = torch.rand(128, 512, 512)
# bigCMatrix = torch.rand(256, 256, 256)

# # 3.2 Функция измерения времени (5 баллов)
# def timeCompleted():
#     startTimeCPU = time.time()
#     startTimeGPU = torch.cuda.Event()


# # 3.3 Сравнение операций (10 баллов)
# # Сравните время выполнения следующих операций на CPU и CUDA:
# # Матричное умножение (torch.matmul)
# startTimeCPU = time.time()
# torch.matmul(bigAMatrix, bigAMatrixAlt``)
# finishTimeCPU = time.time() - startTimeCPU

# print(finishTimeCPU)

import torch
import time
import pandas as pd

# 3.1 Подготовка данных (5 баллов)
# Создание больших матриц
matrices = {
    "64x1024x1024": torch.randn(64, 1024, 1024),
    "128x512x512": torch.randn(128, 512, 512),
    "256x256x256": torch.randn(256, 256, 256),
}

# 3.2 Функция измерения времени (5 баллов)
device_cpu = torch.device("cpu")
device_gpu = torch.device("cuda")

# Функции для измерения времени выполнения операций
def measure_time_cpu(func, *args):
    start = time.time()
    func(*args)
    end = time.time()
    return (end - start) * 1000  # в миллисекундах

def measure_time_gpu(func, *args):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    func(*args)
    end.record()
    
    torch.cuda.synchronize()
    return start.elapsed_time(end)  # в миллисекундах

# 3.3 Сравнение операций (10 баллов)

operations = {
    "Матричное умножение": lambda x: torch.matmul(x, x.transpose(-1, -2)),
    "Поэлементное сложение": lambda x: x + x,
    "Поэлементное умножение": lambda x: x * x,
    "Транспонирование": lambda x: x.transpose(-1, -2),
    "Вычисление суммы всех элементов": lambda x: torch.sum(x),
}

results = []

for name, tensor in matrices.items():
    for op_name, operation in operations.items():
        # CPU
        cpu_time = measure_time_cpu(operation, tensor.to(device_cpu))
        
        # GPU
        if torch.cuda.is_available():
            tensor_gpu = tensor.to(device_gpu)
            gpu_time = measure_time_gpu(operation, tensor_gpu)
            speedup = cpu_time / gpu_time
        else:
            gpu_time = None
            speedup = None
        
        results.append({
            "Размер": name,
            "Операция": op_name,
            "CPU (мс)": round(cpu_time, 2),
            "GPU (мс)": round(gpu_time, 2) if gpu_time else "-",
            "Ускорение": f"{cpu_time / gpu_time:.1f}x" if gpu_time else "-"
        })

# Выводим результаты в виде таблицы
#       Размер               Операция CPU (мс)  GPU (мс)  Ускорение
# 64x1024x1024    Матричное умножение   377.90     54.88       6.9x
# 64x1024x1024  Поэлементное сложение    39.44     10.17       3.9x
# 64x1024x1024 Поэлементное умножение    21.25      5.89       3.6x
# 64x1024x1024       Транспонирование     0.00      0.05       0.0x
# 64x1024x1024        Сумма элементов     6.56      7.39       0.9x
#  128x512x512    Матричное умножение   125.15      6.15      20.3x
#  128x512x512  Поэлементное сложение    12.25      1.56       7.9x
#  128x512x512 Поэлементное умножение     8.09      1.57       5.1x
#  128x512x512       Транспонирование     0.00      0.07       0.0x
#  128x512x512        Сумма элементов     3.54      0.81       4.4x
#  256x256x256    Матричное умножение    25.05      1.76      14.2x
#  256x256x256  Поэлементное сложение     4.11      0.84       4.9x
#  256x256x256 Поэлементное умножение     4.11      0.80       5.1x
#  256x256x256       Транспонирование     0.00      0.04       0.0x
#  256x256x256        Сумма элементов     2.52      0.50       5.0x
df = pd.DataFrame(results)
print(df.to_string(index=False))

# 3.4 Анализ результатов (5 баллов)
# Проанализируйте результаты:
# - Какие операции получают наибольшее ускорение на GPU?
#   Ответ: Матричное умножение, поэлеметное сложение и умножение

# - Почему некоторые операции могут быть медленнее на GPU?
#   Ответ: Их эффективнее делать на CPU, прямо на процессоре
#   не тратя время на передачу данных GPU. Также на маленьких 
#   матрицах CPU быстрее

# - Как размер матриц влияет на ускорение?
#   Ответ: Чем больше матрица, тем больше нужно вычислений и следовательно 
#   больше времени. В итоге получается что ускорение уменьшается

# - Что происходит при передаче данных между CPU и GPU?
#   Ответ: 1) Выделение памяти на GPU 
#   2) Передача данных 