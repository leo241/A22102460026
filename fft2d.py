import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

class Sample:
    def __init__(self):
        # 定义部分本次模型的公共参数
        self.L = 0.0815
        self.N_a = 86
        self.c = 3e8
        self.gamma = 78.986e12
        self.f0 = 78.8e9
        self.T_s = 1.25e-7
    def sample(self,objects):
        L = self.L
        N_a = self.N_a
        c = self.c
        gamma = self.gamma
        f0 = self.f0
        T_s = self.T_s
        line_x = [-L / 2 + n * L / (N_a - 1) for n in range(N_a)]  # 天线的横坐标
        line_real = list()
        for line in line_x:  # 遍历每个天线
            line_t = list()
            for t in range(256):  # 遍历每个采样时间
                real_sum = 0  # 实数部分所有物体的叠加
                img_sum = 0
                for object in objects:
                    r, theta, a_k = object
                    x = r * np.sin(theta)
                    y = r * np.cos(theta)
                    R_nk = 2 * np.sqrt(np.square(line - x) + np.square(y))
                    real_sum += a_k * np.cos(2 * np.pi * gamma * T_s * t * R_nk / c + 2 * np.pi * f0 * R_nk / c)
                    img_sum += a_k * np.sin(2 * np.pi * gamma * T_s * t * R_nk / c + 2 * np.pi * f0 * R_nk / c)
                line_t.append(real_sum + np.complex128('j') * img_sum)
            line_real.append(line_t)
        return np.asarray(line_real)
    def sample_with_variance(self,objects,variance_x,variance_y):
        L = self.L
        N_a = self.N_a
        c = self.c
        gamma = self.gamma
        f0 = self.f0
        T_s = self.T_s
        delta_x = np.random.normal(0,variance_x)
        delta_y = np.random.normal(0,variance_y)
        line_x = [-L / 2 + n * L / (N_a - 1) for n in range(N_a)]  + delta_x # 天线的横坐标
        line_real = list()
        for line in line_x:  # 遍历每个天线
            line_t = list()
            for t in range(256):  # 遍历每个采样时间
                real_sum = 0  # 实数部分所有物体的叠加
                img_sum = 0
                for object in objects:
                    r, theta, a_k = object
                    x = r * np.sin(theta)
                    y = r * np.cos(theta)
                    R_nk = 2 * np.sqrt(np.square(line - x) + np.square(y + delta_y))
                    real_sum += a_k * np.cos(2 * np.pi * gamma * T_s * t * R_nk / c + 2 * np.pi * f0 * R_nk / c)
                    img_sum += a_k * np.sin(2 * np.pi * gamma * T_s * t * R_nk / c + 2 * np.pi * f0 * R_nk / c)
                line_t.append(real_sum + np.complex128('j') * img_sum)
            line_real.append(line_t)
        return np.asarray(line_real)
    def sample_with_rvar(self,objects,variance_r):
        L = self.L
        N_a = self.N_a
        c = self.c
        gamma = self.gamma
        f0 = self.f0
        T_s = self.T_s
        delta_r = np.random.normal(0,variance_r)
        theta0 = np.random.uniform(0,2*np.pi)
        line_x = [-L / 2 + n * L / (N_a - 1) for n in range(N_a)]  + delta_r*cos(theta0) # 天线的横坐标
        line_real = list()
        for line in line_x:  # 遍历每个天线
            line_t = list()
            for t in range(256):  # 遍历每个采样时间
                real_sum = 0  # 实数部分所有物体的叠加
                img_sum = 0
                for object in objects:
                    r, theta, a_k = object
                    x = r * np.sin(theta)
                    y = r * np.cos(theta)
                    R_nk = 2 * np.sqrt(np.square(line - x) + np.square(y + delta_r*sin(theta0)))
                    real_sum += a_k * np.cos(2 * np.pi * gamma * T_s * t * R_nk / c + 2 * np.pi * f0 * R_nk / c)
                    img_sum += a_k * np.sin(2 * np.pi * gamma * T_s * t * R_nk / c + 2 * np.pi * f0 * R_nk / c)
                line_t.append(real_sum + np.complex128('j') * img_sum)
            line_real.append(line_t)
        return np.asarray(line_real)
    def sample_with_noise(self,objects,variance = 0.1):
        er = np.random.normal(0, variance, size=(86, 256))  # 添加指定方差的白噪声，默认为0.1
        return  self.sample(objects) + er
    def random_n_sample(self,n):
        objects = list()
        for i in range(n):
            r = np.random.uniform(0, 10)
            theta = np.random.uniform(-5 / 18 * np.pi, 5 / 18 * np.pi) # 开口张开100°
            ak = np.abs(np.random.normal(0.5, 1)) + 1e3
            objects.append((r, theta, ak))
        line_real = self.sample(objects)
        return line_real, np.asarray(objects)
    def random_n_sample_with_noise(self,n,variance):
        er = np.random.normal(0, variance, size=(86, 256))  # 添加指定方差的白噪声，默认为0.1
        line_real,objects = self.random_n_sample(n)
        return line_real + er, objects


def find_max_id(mat):
    id = np.where(mat == np.max(mat))
    theta_id =np.asarray(id).tolist()[0][0]
    R_id = np.asarray(id).tolist()[1][0]
    return R_id,theta_id

def find_second_id(mat):
    mat2 = deepcopy(mat)
    y,x = find_max_id(mat2)
    mat2[x][y] = -np.inf
    return find_max_id(mat2)

def find_n_id(mat,n):
    mat2 = deepcopy(mat)
    for i in range(n-1):
        y,x = find_max_id(mat2)
        mat2[x][y] = -np.inf
    return find_max_id(mat2)

def top_n_id(mat,n):
    results = list()
    for i in range(n):
        results.append(find_n_id(mat, i + 1))
    return results

def is_wave(mat,i,j,threshold):
    if mat[i,j] > max(mat[i-1,j],mat[i+1,j],mat[i,j-1],mat[i,j+1],mat[i+1,j+1],mat[i+1,j-1],mat[i-1,j+1],mat[i-1,j-1],threshold):
        return True

def find_wave(mat,threshold):
    wave_id_list = list()
    wave_list = list()
    results = list()
    height,length = mat.shape
    for i in np.arange(height-2) + 1:
        for j in np.arange(length-2)+1:
            if is_wave(mat,i,j,threshold):
                wave_id_list.append([i,j])
                wave_list.append(mat[i][j])
    # 下面找wave中前n个最大的，对应的id返回
    b = np.argsort(wave_list)[::-1]
    for k in range(len(wave_list)):
        results.append(wave_id_list[b[k]])
    return results

def FFT2d(X):
    ff = np.fft.fft2(X, norm=None)
    ff_up = ff[0:43, :]
    ff_down = ff[43:, :]
    ff2 = np.r_[ff_down, ff_up]
    # print(ff2.shape)
    # ff2 = ff
    plt.imshow(np.abs(ff2))
    plt.savefig('FFt-2d')
    plt.show()
    return ff2

def id2R(id):
    L = 0.0815
    N_a = 86
    c = 3e8
    gamma = 78.986e12
    f0 = 78.8e9
    T_s = 1.25e-7
    d = L / (N_a - 1)
    freq = np.fft.fftfreq(256, 1)

    for i in range(len(freq)):
        if freq[i] < 0:
            freq[i] = 1 + freq[i]
    down = 2 * np.pi * gamma * T_s
    freq = freq * c / down * np.pi
    freq = np.sort(freq)
    return freq[id]

# def id2theta(id):
#     L = 0.0815
#     N_a = 86
#     c = 3e8
#     gamma = 78.986e12
#     f0 = 78.8e9
#     T_s = 1.25e-7
#     d = L / (N_a - 1)
#     x_axis = np.arange(N_a)
#     for i in range(len(x_axis)):
#         if x_axis[i] > 43:
#             x_axis[i] = x_axis[i] - 86
#     x_axis = -np.arcsin(c / f0 / d / 2 * x_axis / 86) / np.pi * 180
#     x_axis = np.sort(x_axis)[::-1]
#     return x_axis[id]

def id2theta(id):
    L = 0.0815
    N_a = 86
    c = 3e8
    gamma = 78.986e12
    f0 = 78.8e9
    T_s = 1.25e-7
    d = L / (N_a - 1)
    # x_axis = np.arange(N_a)
    # for i in range(len(x_axis)):
    #     if x_axis[i] > 43:
    #         x_axis[i] = x_axis[i] - 86
    # x_axis = -np.arcsin(c / f0 / d / 2 * x_axis / 86) / np.pi * 180
    # x_axis = np.sort(x_axis)[::-1]
    id = 43 - id
    theta = np.arcsin(
        c / f0 / d / 2 * id / 86 ) / np.pi * 180
    return theta
    # return x_axis[id]

def diag2(nita,  mu,mine):
    my_diag = np.zeros((86, 86))
    my_diag = np.complex128(my_diag)  # 将矩阵转成复数矩阵，否则会自动将复数部分砍掉
    for i in range(86):
        # my_diag[i][i] = np.exp(np.complex128('j') * i * nita)
        my_diag[i][i] = np.cos(i * nita) + np.sin(i * nita) * np.complex128('j')

    my_diag2 = np.zeros((256, 256))
    my_diag2 = np.complex128(my_diag2)  # 将矩阵转成复数矩阵，否则会自动将复数部分砍掉
    for i in range(256):
        # my_diag2[i][i] = np.exp(np.complex128('j') * i * mu)
        my_diag2[i][i] = np.cos(i * mu) + np.sin(i * mu) * np.complex128('j')
    return my_diag @ mine @ my_diag2

def find_neighbor(target_id):
    i, j = target_id
    return [[i+1,j+1],[i+1,j],[i-1,j],[i,j+1],[i,j-1],[i+1,j-1],[i-1,j+1],[i-1,j-1]]


def adjust_angle2(mine, threshold):
    '''我们的目的就是要对id_combination进行修正'''
    ff2 = FFT2d(X)
    top_id = find_wave(ff2, threshold=threshold) # 找到峰值的id组合
    # stat4 = self.FFT_for_angle_id(mine, threshold=threshold)
    # id_combination = list(stat4[0][0])
    star_num = len(top_id)  # 物体的数量

    wave_star = [-np.inf for i in range(star_num)]
    angle_mu_star = [None for i in range(star_num)]
    a_matrix_star = [None for i in range(star_num)]
    # id_adjust = deepcopy(id_combination)
    # raw = self.raw

    angle_grid = np.linspace(-np.pi / 86, np.pi / 86, 50)
    angle_gridmiu = np.linspace(-np.pi / 256, np.pi / 256, 50)
    for star_id in range(star_num):
        max_wave_list = list()
        for angle in angle_grid:
            for anglemiu in angle_gridmiu:
                target_id = top_id[star_id]  # 在这个的±1搜索，这里不需要处理边界条件
                a_matrix = diag2(angle,anglemiu, mine) # 待作fft的矩阵

                ff = np.fft.fft2(a_matrix, norm=None)

                neighbors = find_neighbor(target_id)
                if target_id == 0:
                    neighbors.remove(-1)
                elif target_id == 85:
                    neighbors.remove(86)
                neighbors_wave = [ff[neighbor[0]][neighbor[1]] for neighbor in neighbors]
                max_wave = max(neighbors_wave)
                max_wave_list.append(max_wave)
                if max_wave > wave_star[star_id]:
                    wave_star[star_id] = max_wave
                    # id_adjust[star_id] = neighbors[neighbors_wave.index(max_wave)]  # 如果产生了某个id的更大值，进行修正
                    angle_mu_star[star_id] = [angle,anglemiu]
                    a_matrix_star[star_id] = a_matrix
        # plt.plot(angle_grid, max_wave_list)
        # plt.savefig('phi_wave.png')
        # plt.show()
    # mag_frames = [np.absolute(np.fft.fft(i, 86)) for i in a_matrix_star]
    return np.asarray(angle_mu_star)

if __name__ == '__main__':
    L = 0.0815
    N_a = 86
    c = 3e8
    gamma = 78.986e12
    f0 = 78.8e9
    T_s = 1.25e-7
    d = L / (N_a - 1)

    X = np.load("data_q3.npy")[0,:,:]
    sample_tool = Sample()
    objects = np.asarray([(4.5,0.25*np.pi, 2),(4.5,-0.25*np.pi, 2),(9.7,-0.1*np.pi, 2)])
    # objects = np.asarray([(4, 0, 5)])
    # objects = np.asarray([(6,0,1)])
    X = sample_tool.sample(objects)

    threshold = 1e3
    ff2 = FFT2d(X)
    top_id = find_wave(ff2,threshold=threshold)
    for id_combination in top_id:
        theta_id, R_id  = id_combination
        print('R:',id2R(R_id))
        print('theta:', id2theta(theta_id))
        print('\n')

    angle_mu_star = adjust_angle2(X,threshold=threshold) # 得到最优修正
    for iter in range(len(top_id)):
        theta_id, R_id = top_id[iter]
        angle_star, mu_star = angle_mu_star[iter]
        # theta_id = 85 - theta_id

        theta_id = 43 - theta_id
        theta_adjust = np.arcsin(
            c / f0 / d / 2 * theta_id / 86 - angle_star * c / f0 / d / 4 / np.pi) / np.pi * 180
        R_adjust = id2R(R_id) - c*mu_star/4/np.pi/gamma/T_s
        print('\n')
        print('R-adjust:',R_adjust)
        print('theta-adjust:',theta_adjust)


    L = 0.0815
    N_a = 86
    c = 3e8
    gamma = 78.986e12
    f0 = 78.8e9
    T_s = 1.25e-7
    d = L / (N_a - 1)

    x_axis = np.arange(N_a)
    for i in range(len(x_axis)):
        if x_axis[i] > 43:
            x_axis[i] = x_axis[i] - 86
    x_axis = -np.arcsin(c / f0 / d / 2 * x_axis / 86) / np.pi * 180
    x_axis = np.sort(x_axis)




    freq = np.fft.fftfreq(256, 1)


    # for i in range(len(freq)):
    #     if freq[i] < 0:
    #         freq[i] = 1 + freq[i]
    # down = 2 * np.pi * gamma * T_s
    # freq = freq * c / down * np.pi
    # freq = np.sort(freq)

    # plt.yticks(np.arange(0,86,32),np.arange(x_axis[0],x_axis[-1],32))
