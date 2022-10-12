from sample import *
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def sample(objects):
    L = 0.0815
    N_a = 86
    c = 3e8
    gamma = 78.986e12
    f0 = 78.8e9
    T_s = 1.25e-7
    line_x = [-L/2 + n * L/(N_a-1) for n in range(N_a)] # 天线的横坐标
    line_real = list()
    for line in line_x: # 遍历每个天线
        line_t = list()
        for t in range(256): # 遍历每个采样时间
            real_sum = 0 # 实数部分所有物体的叠加
            for object in objects:
                r,theta,a_k = object
                x = r * np.sin(theta)
                y = r * np.cos(theta)
                R_nk = 2 * np.sqrt(np.square(line - x) + np.square(y))
                real_sum += a_k * np.cos(2 * np.pi * gamma * T_s * t * R_nk/c + 2 * np.pi * f0 *R_nk/c)
            line_t.append(real_sum)
        line_real.append(line_t)
    return np.asarray(line_real)

# 对2d-music算法的a向量

def tao(m,theta,d):
    L = 0.0815
    N_a = 86
    c = 3e8
    return 2/c * (d + m * L/(N_a - 1) * np.sin(theta))

def a_vec(theta,d):
    L = 0.0815
    N_a = 86
    c = 3e8
    # gamma = 78.986e12
    f0 = 78.8e9
    # T_s = 1.25e-7

    y_k = -2 * np.pi * (-2 * L/(N_a - 1) * np.sin(theta) * f0/c)
    phi_k = np.pi * (4 * np.square(L/(N_a - 1)) * np.square(np.cos(theta))*f0/c/d)
    a_vec = [np.cos(i * y_k + np.square(i) * phi_k) + np.sin(i * y_k + np.square(i) * phi_k) * np.complex128('j') for i in range(N_a)]
    return np.asarray(a_vec)

def a_vec2(theta,d):
    L = 0.0815
    N_a = 86
    c = 3e8
    gamma = 78.986e12
    f0 = 78.8e9
    T_s = 1.25e-7
    a_vec = [np.exp(np.complex128('j') * 2 * np.pi* (f0 * tao(0,theta,d) + gamma * tao(i,theta,d) * 1/T_s)) for i in range(N_a)]
    return np.asarray(a_vec)

def music_2d(a_vec, X):
    J = 256
    R = np.matmul(X,X.conj().T)/J
    eig,vec = np.linalg.eig(R)
    E_n = vec[:,1:] # 取噪声对应的特征向量
    P = 1/(a_vec.conj().T @ E_n @ E_n.conj().T @ a_vec)
    return P



# 定义搜索网格
theta_grid = np.linspace(-np.pi * 5/18, np.pi * 5/18,20) # 对角度进行网格搜索
d_grid = np.linspace(1,10,20) # 对距离进行网格搜索
# X = np.load("data_q3.npy")[30]
X = np.load("data_q1.npy")
# X = random_n_sample(5)[0]
# X = np.load("data_q1.npy")
# X, object = random_n_sample(1)
# X = X[:,]
object1 = np.asarray([(1,0.15 * np.pi, 0.5)])

X = sample_complex(object1)


P_list = list()
max_P = -np.inf
d_star = None
theta_star = None
for theta in tqdm(theta_grid):
    for d in d_grid:
        a_vec1 = a_vec2(theta,d)
        P_list.append(music_2d(a_vec1,X))
        if music_2d(a_vec1,X) > max_P:
            max_P = music_2d(a_vec1,X)

            d_star = d
            theta_star = theta

print(d_star, theta_star)
plt.plot(P_list)
plt.show()
