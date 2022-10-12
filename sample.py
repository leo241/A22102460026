import numpy as np

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
                img_sum += a_k * np.sin(2 * np.pi * gamma * T_s * t * R_nk / c + 2 * np.pi * f0 * R_nk / c)
            line_t.append(real_sum + np.complex128('j') * img_sum)
        line_real.append(line_t)
    return np.asarray(line_real)

def random_sample():
    r1 = np.random.uniform(0, 10)
    r2 = np.random.uniform(0, 10)
    theta1 = np.random.uniform(2 / 9 * np.pi, 7 / 9 * np.pi)
    theta2 = np.random.uniform(2 / 9 * np.pi, 7 / 9 * np.pi)
    object1 = (r1, theta1, np.abs(np.random.normal(0, 1)))
    object2 = (r2, theta2, np.abs(np.random.normal(0, 1)))
    objects = [object1, object2]
    line_real = sample(objects)
    return line_real

def random_n_sample(n):
    # r1 = np.random.uniform(0, 10)
    # r2 = np.random.uniform(0, 10)
    # theta1 = np.random.uniform(2 / 9 * np.pi, 7 / 9 * np.pi)
    # theta2 = np.random.uniform(2 / 9 * np.pi, 7 / 9 * np.pi)
    # object1 = (r1, theta1, np.abs(np.random.normal(0, 1)))
    # object2 = (r2, theta2, np.abs(np.random.normal(0, 1)))
    # objects = [object1, object2]
    objects = list()
    for i in range(n):
        r = np.random.uniform(0, 10)
        theta = np.random.uniform(-5 / 18 * np.pi, 5 / 18 * np.pi)
        ak = np.abs(np.random.normal(0, 1))
        objects.append((r,theta,ak))
    line_real = sample(objects)
    objects = np.asarray(objects)
    objects = objects[np.argsort(objects[:, 0])] # 这行代码很关键，让距离从大到小排列
    return line_real, objects

if __name__ == '__main__':
    object1 = (1,np.pi/4,1)
    object2 = (1,np.pi * 3/4,0.25)
    objects = [object1,object2]
    line_real = sample(objects)
    given_real = np.real(np.load('data_q1.npy'))

