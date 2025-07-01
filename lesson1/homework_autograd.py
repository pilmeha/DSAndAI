import torch

# Домашнее задание к уроку 1: Основы PyTorch

# Задание 2: Автоматическое дифференцирование (25 баллов)

# 2.1 Простые вычисления с градиентами (8 баллов)

# Создайте тензоры x, y, z с requires_grad=True
x = torch.rand(3, requires_grad=True)
y = torch.rand(3, requires_grad=True)
z = torch.rand(3, requires_grad=True)

# Вычислите функцию: f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
funcXYZ = x ** 2 + y ** 2 + z ** 2 + 2 * x * y * z

# Найдите градиенты по всем переменным
funcXYZ.sum().backward()

dx = x.grad
dy = y.grad
dz = z.grad

# Проверьте результат аналитически
def analitical_gradients(x, y, z):
    a_dx = 2 * x + 2 * y * z
    a_dy = 2 * y + 2 * x * z
    a_dz = 2 * z + 2 * x * y
    return a_dx, a_dy, a_dz

a_dx, a_dy, a_dz = analitical_gradients(x.detach(), y.detach(), z.detach())


# 2.2 Градиент функции потерь (9 баллов)
# def mseDef(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
#     return ((y_pred - y_true) ** 2).mean()

#     # loss.backward()
#     # w_grad = mse_w.grad
#     # b_grad = mse_b.grad

#     # mse_w.grad.zero_()
#     # mse_b.grad.zero_()

#     # return 


# mse_x = torch.rand(4)
# mse_y = torch.rand(4)

# mse_w = torch.rand(1, requires_grad=True)
# mse_b = torch.rand(1, requires_grad=True)

# y_pred = mse_w * mse_x + mse_b

# loss = mseDef(y_pred, mse_y)

# loss.backward()
# w_grad = mse_w.grad
# b_grad = mse_b.grad

# mse_w.grad.zero_()
# mse_b.grad.zero_()

def mse_loss(w: torch.Tensor, b: torch.Tensor, x: torch.Tensor, y_true: torch.Tensor):
    # Проверка размерностей
    assert x.shape == y_true.shape
    
    # Прямой проход (forward pass)
    n = x.shape[0]
    y_pred = w * x + b
    loss = ((y_pred - y_true)**2).mean()
    
    # Обратный проход (backward pass)
    loss.backward()
    
    # Собираем градиенты
    grads = {
        'dw': w.grad,
        'db': b.grad
    }
    
    # Обнуляем градиенты для следующих вычислений
    w.grad.zero_()
    b.grad.zero_()
    
    return loss.item(), grads

mse_x = torch.rand(4)
mse_y_true = torch.rand(4)

mse_w = torch.rand(1, requires_grad=True)
mse_b = torch.rand(1, requires_grad=True)

loss, grads = mse_loss(mse_w, mse_b, mse_x, mse_y_true)

# 2.3 Цепное правило (8 баллов)

# Реализуйте составную функцию: f(x) = sin(x^2 + 1)
tensor_x_sin = torch.rand(3, requires_grad=True)
f = torch.sin(tensor_x_sin ** 2 + 1)

# Найдите градиент df/dx
f.sum().backward(retain_graph=True)
grad_x_backward = tensor_x_sin.grad

# Проверьте результат с помощью torch.autograd.grad
grad_x_autograd = torch.autograd.grad(f.sum(), tensor_x_sin)
