import random


'''
梯度下降算法是为了在函数空间内找到极限值点
算法要点：
1. 目标函数
2. 导数
3. 学习率（步长）
4. 最大迭代次数、误差率
5. 使得目标函数最小化的theta向量（系数？），以及极值点
'''


# http://yphuang.github.io/blog/2016/03/17/Gradient-Descent-Algorithm-Implementation-in-Python/

# 定义梯度估计函数
def estimate_gradient(f, v, h=0.00001):
    return [partial_difference_quotient(f, v, i, h)
            for i, _ in enumerate(v)]


# 定义偏导估计函数
def partial_difference_quotient(f, v, i, h):
    w = [v_j + (h if j == i else 0)
         for j, v_j in enumerate(v)]
    return (f(w) - f(v)) / h


# 向量的减
def vector_substract(v, w):
    return [v_i - w_i for v_i, w_i in zip(v, w)]


# 点乘
def dot(v, w):
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


# 向量之间的距离
def distance(v, w):
    return sum_of_squares(vector_substract(v, w))


# 定义步长函数
def step(v, direction, step_size):
    return [v_i + step_size * direction_i
            for v_i, direction_i in zip(v, direction)]


# 向量的平房和
def sum_of_squares(v):
    return dot(v, v)


# 定义梯度
def sum_of_squares_gradient(v):
    return [2 * v_i for v_i in v]


# 防止输入的自变量超过定义域范围，超过范围的定义为inf
def safe(f):
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf')
    return safe_f


# 使用梯度下降来找到最小化目标函数的theta
def minimize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

    theta = theta_0
    target_fn = safe(target_fn)
    value = target_fn(theta)

    while True:
        gradient = gradient_fn(theta)
        next_thetas = [step(theta, gradient, -step_size)
                       for step_size in step_sizes]

    # 选择最小化目标函数的theta
    next_theta = min(next_thetas, key=target_fn)
    next_value = target_fn(next_theta)

    # 停止准则
    if abs((value - next_value)/value) < tolerance:
        return theta, value

    else:
        theta, value = next_theta, next_value


# 测试
max_iter = 1000
iter = 1
theta_0 = [random.randint(-10, 10) for i in range(3)]

while True:
    theta, value = minimize_batch(
        target_fn=sum_of_squares,
        gradient_fn=sum_of_squares_gradient,
        theta_0=theta_0, tolerance=0.0001)
    if (iter < max_iter) or (value == sum_of_squares(theta_0)):
        break
    theta_0 = theta
    iter += 1


print(theta, iter)
