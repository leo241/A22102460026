import numpy as np
from numpy import *
from matplotlib import pyplot as plt
from collections import Counter
from copy import deepcopy

class Sample: # 数值模拟类别封装
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

class FFT:
    def __init__(self,raw = 1):
        # 定义部分本次模型的公共参数
        self.L = 0.0815
        self.N_a = 86
        self.c = 3e8
        self.gamma = 78.986e12
        self.f0 = 78.8e9
        self.T_s = 1.25e-7
        self.raw = raw # 在FFT时默认使用哪一行的数据

    def FFT_for_distance(self,frames0):
        raw = self.raw
        c = self.c
        gamma = self.gamma
        T_s = self.T_s
        frames = frames0
        NFFT = 256
        mag_frames = np.absolute(np.fft.fft(frames, NFFT))  # Magnitude of the FFT
        freq = np.fft.fftfreq(NFFT,1)
        for i in range(len(freq)):
            if freq[i] <0 :
                freq[i] = 1+freq[i]
        down = 2 * np.pi * gamma * T_s
        freq = freq * c / down * np.pi
        plt.figure(figsize=(11, 4), dpi=500)
        plt.plot(freq, mag_frames[raw, :], color='blue')
        plt.xlabel("frequency", fontsize=14)
        plt.ylabel("Amplitude", fontsize=14)
        plt.title("FFT", fontsize=14)
        plt.savefig(f'fft_distance.png')
        plt.show()

        return mag_frames
    def FFT_for_distance_star(self,frames0,threshold = 1):
        raw = self.raw
        c = self.c
        gamma = self.gamma
        T_s = self.T_s
        frames = frames0

        NFFT = 256
        mag_frames = np.absolute(np.fft.fft(frames, NFFT))  # Magnitude of the FFT
        freq = np.fft.fftfreq(NFFT, 1)
        for i in range(len(freq)):
            if freq[i] < 0:
                freq[i] = 1 + freq[i]
        down = 2 * np.pi * gamma * T_s
        freq = freq * c / down * np.pi

        sort_id = np.argsort(freq)
        freq = np.sort(freq)
        mag_frames = mag_frames[:,sort_id]

        R_list = list()
        length = mag_frames.shape[0]
        for i in range(length):  # 遍历所有天线
            line_list = list()
            for j in np.arange(mag_frames.shape[1] - 2) + 1:
                if mag_frames[i][j] > mag_frames[i][j - 1] and mag_frames[i][j] > mag_frames[i][j + 1] and \
                        mag_frames[i][j] > threshold:
                    line_list.append(freq[j])
            R_list.append(frozenset(line_list))
        stat = Counter(R_list).most_common(3)
        return stat

    def FFT_for_angle(self,frames0):
        frames0 = frames0.T # 在做角度时需要转置为256 * 86
        raw = self.raw
        c = self.c
        gamma = self.gamma
        L = self.L
        T_s = self.T_s
        N_a = self.N_a
        frames = frames0
        NFFT = 86
        d = L / (N_a - 1)
        f0 = self.f0
        mag_frames = np.absolute(np.fft.fft(frames, NFFT))  # Magnitude of the FFT
        plt.figure(figsize=(11, 4), dpi=500)
        x_axis = np.arange(NFFT)
        for i in range(len(x_axis)):
            if x_axis[i] > 43:
                x_axis[i] = x_axis[i] - 86
        x_axis = -np.arcsin(c / f0 / d / 2 * x_axis / 86) / np.pi * 180

        sort_id = np.argsort(x_axis)
        x_axis = np.sort(x_axis)
        mag_frames = mag_frames[:, sort_id]

        plt.figure(figsize=(11, 4), dpi=500)
        plt.plot(x_axis, mag_frames[raw, :], color='blue')
        plt.xlabel("angle", fontsize=14)
        plt.ylabel("Amplitude", fontsize=14)
        plt.title("FFT-angle", fontsize=14)
        plt.savefig(f'fft_angle.png')
        plt.show()



    def FFT_for_angle_star(self,frames0,threshold = 20):
        frames0 = frames0.T # 在做角度时需要转置为256 * 86
        raw = self.raw
        c = self.c
        gamma = self.gamma
        L = self.L
        T_s = self.T_s
        N_a = self.N_a
        frames = frames0
        NFFT = 86
        d = L / (N_a - 1)
        f0 = self.f0
        mag_frames = np.absolute(np.fft.fft(frames, NFFT))  # Magnitude of the FFT

        x_axis = np.arange(NFFT)
        for i in range(len(x_axis)):
            if x_axis[i] > 43:
                x_axis[i] = x_axis[i] - 86
        x_axis = -np.arcsin(c / f0 / d / 2 * x_axis / 86) / np.pi * 180

        sort_id = np.argsort(x_axis)
        x_axis = np.sort(x_axis)
        mag_frames = mag_frames[:, sort_id]

        R_list = list()
        length = mag_frames.shape[0]
        for i in range(length):  # 遍历所有天线
            line_list = list()
            for j in np.arange(mag_frames.shape[1] - 2) + 1:
                if mag_frames[i][j] > mag_frames[i][j - 1] and mag_frames[i][j] > mag_frames[i][j + 1] and \
                        mag_frames[i][j] > threshold:
                    line_list.append(x_axis[j])
            R_list.append(frozenset(line_list))
        stat = Counter(R_list).most_common(3)
        return stat


    def FFT_for_angle_id(self,frames0,threshold = 20):
        frames0 = frames0.T # 在做角度时需要转置为256 * 86
        raw = self.raw
        c = self.c
        gamma = self.gamma
        L = self.L
        T_s = self.T_s
        N_a = self.N_a
        frames = frames0
        NFFT = 86
        d = L / (N_a - 1)
        f0 = self.f0
        mag_frames = np.absolute(np.fft.fft(frames, NFFT))  # Magnitude of the FFT

        x_axis = np.arange(NFFT)
        # for i in range(len(x_axis)):
        #     if x_axis[i] > 43:
        #         x_axis[i] = x_axis[i] - 86
        # x_axis = -np.arcsin(c / f0 / d / 2 * x_axis / 86) / np.pi * 180

        sort_id = np.argsort(x_axis)
        x_axis = np.sort(x_axis)
        mag_frames = mag_frames[:, sort_id]

        R_list = list()
        length = mag_frames.shape[0]
        for i in range(length):  # 遍历所有天线
            line_list = list()
            if mag_frames[i][0] > mag_frames[i][85] and mag_frames[i][0] > mag_frames[i][1] and \
                    mag_frames[i][0] > threshold: # 处理j = 0的情况
                line_list.append(x_axis[0])
            for j in np.arange(mag_frames.shape[1] - 2) + 1:
                if mag_frames[i][j] > mag_frames[i][j - 1] and mag_frames[i][j] > mag_frames[i][j + 1] and \
                        mag_frames[i][j] > threshold:
                    line_list.append(x_axis[j])
            if mag_frames[i][85] > mag_frames[i][84] and mag_frames[i][85] > mag_frames[i][0] and \
                    mag_frames[i][85] > threshold: # 处理j = 85的情况
                line_list.append(x_axis[85])
            R_list.append(frozenset(line_list))
        stat = Counter(R_list).most_common(3)
        return stat

    def FFT_for_distance_id(self,frames0,threshold = 20):
        frames0 = frames0 # 在做角度时需要转置为256 * 86
        raw = self.raw
        c = self.c
        gamma = self.gamma
        L = self.L
        T_s = self.T_s
        N_a = self.N_a
        frames = frames0
        NFFT = 256
        d = L / (N_a - 1)
        f0 = self.f0
        mag_frames = np.absolute(np.fft.fft(frames, NFFT))  # Magnitude of the FFT

        x_axis = np.arange(256)
        # for i in range(len(x_axis)):
        #     if x_axis[i] > 43:
        #         x_axis[i] = x_axis[i] - 86
        # x_axis = -np.arcsin(c / f0 / d / 2 * x_axis / 86) / np.pi * 180

        sort_id = np.argsort(x_axis)
        x_axis = np.sort(x_axis)
        mag_frames = mag_frames[:, sort_id]

        R_list = list()
        length = mag_frames.shape[0]
        for i in range(length):  # 遍历所有天线
            line_list = list()
            if mag_frames[i][0] > mag_frames[i][255] and mag_frames[i][0] > mag_frames[i][1] and \
                    mag_frames[i][0] > threshold: # 处理j = 0的情况
                line_list.append(x_axis[0])
            for j in np.arange(mag_frames.shape[1] - 2) + 1:
                if mag_frames[i][j] > mag_frames[i][j - 1] and mag_frames[i][j] > mag_frames[i][j + 1] and \
                        mag_frames[i][j] > threshold:
                    line_list.append(x_axis[j])
            if mag_frames[i][255] > mag_frames[i][254] and mag_frames[i][255] > mag_frames[i][0] and \
                    mag_frames[i][255] > threshold: # 处理j = 85的情况
                line_list.append(x_axis[255])
            R_list.append(frozenset(line_list))
        stat = Counter(R_list).most_common(3)
        return stat


    def diag2(self,nita, mine):
        my_diag = np.zeros((86, 86))
        my_diag = np.complex128(my_diag)  # 将矩阵转成复数矩阵，否则会自动将复数部分砍掉
        for i in range(86):
            # my_diag[i][i] = np.exp(np.complex128('j') * i * nita)
            my_diag[i][i] = np.cos(i * nita) + np.sin(i * nita) * np.complex128('j')
        return my_diag @ mine

    def adjust_angle2(self,mine,threshold = 20):
        '''我们的目的就是要对id_combination进行修正'''
        stat4 = self.FFT_for_angle_id(mine, threshold=threshold)
        id_combination = list(stat4[0][0])
        star_num = len(id_combination) # 物体的数量

        raw_id_star = [None for i in range(star_num)]
        wave_star = [-np.inf for i in range(star_num)]
        angle_star = [None for i in range(star_num)]
        a_matrix_star = [None for i in range(star_num)]
        id_adjust = deepcopy(id_combination)
        raw = self.raw

        angle_grid = np.linspace(-np.pi / 86, np.pi / 86, 50)
        for star_id in range(star_num):
            max_wave_list = list()
            for angle in angle_grid:
                target_id = id_combination[star_id] # 在这个的±1搜索，注意处理0和85的边界条件
                a_matrix = self.diag2(angle, mine).T  # 待作fft的矩阵

                mag_frames = np.absolute(np.fft.fft(a_matrix, 86))

                neighbors = [target_id-1, target_id,target_id + 1]
                if target_id == 0:
                    neighbors.remove(-1)
                elif target_id == 85:
                    neighbors.remove(86)
                raw_mag_frames = mag_frames[raw, :]
                neighbors_wave = [raw_mag_frames[neighbor] for neighbor in neighbors]
                max_wave = max(neighbors_wave)
                max_wave_list.append(max_wave)
                if max_wave > wave_star[star_id]:
                    wave_star[star_id] = max_wave
                    id_adjust[star_id] = neighbors[neighbors_wave.index(max_wave)] # 如果产生了某个id的更大值，进行修正
                    angle_star[star_id] = angle
                    a_matrix_star[star_id] = a_matrix
            plt.plot(angle_grid,max_wave_list)
            plt.savefig('phi_wave.png')
            plt.show()
        mag_frames = [np.absolute(np.fft.fft(i, 86)) for i in a_matrix_star ]
        return np.asarray(angle_star),id_adjust,mag_frames


    def adjust5(self,X,threshold = 20):
        c = self.c
        gamma = self.gamma
        L = self.L
        T_s = self.T_s
        N_a = self.N_a
        d = L / (N_a - 1)
        f0 = self.f0

        angle_star, id_adjust, mag_frames = self.adjust_angle2(X, threshold=threshold)
        stat4 = self.FFT_for_angle_id(X, threshold=threshold)
        id_combination = list(stat4[0][0]) # 原来的ID
        for i in range(len(id_combination)):
            if id_combination[i] > 43:
                id_combination[i] = id_combination[i] - 86
            id_adjust[i] = -np.arcsin(c / f0 / d / 2 * id_combination[i] / 86 - angle_star[i]*c/f0/d/4/np.pi) / np.pi * 180
        # for i in range(len(id_combination)):
        #     comang = c / f0 / d / 2 * id_combination[i] / 86 - angle_star[i] * c / f0 / d / 4 / np.pi
        #     if comang > 1:
        #         comang = comang-2
        #     id_adjust[i] = -np.arcsin(comang) / np.pi * 180
        return id_adjust

    def diag3(self,nita, mine):
        my_diag = np.zeros((256, 256))
        my_diag = np.complex128(my_diag)  # 将矩阵转成复数矩阵，否则会自动将复数部分砍掉
        for i in range(256):
            # my_diag[i][i] = np.exp(np.complex128('j') * i * nita)
            my_diag[i][i] = np.cos(i * nita) + np.sin(i * nita) * np.complex128('j')
        return mine @ my_diag

    def adjust_angle3(self,mine,threshold = 20):
        '''我们的目的就是要对id_combination进行修正'''
        stat4 = self.FFT_for_distance_id(mine, threshold=threshold)
        id_combination = list(stat4[0][0])
        star_num = len(id_combination) # 物体的数量

        raw_id_star = [None for i in range(star_num)]
        wave_star = [-np.inf for i in range(star_num)]
        angle_star = [None for i in range(star_num)]
        a_matrix_star = [None for i in range(star_num)]
        id_adjust = deepcopy(id_combination)
        raw = self.raw

        angle_grid = np.linspace(-np.pi / 256, np.pi / 256, 50)
        for star_id in range(star_num):
            max_wave_list = list()
            for angle in angle_grid:
                target_id = id_combination[star_id] # 在这个的±1搜索，注意处理0和85的边界条件
                a_matrix = self.diag2(angle, mine) # 待作fft的矩阵

                mag_frames = np.absolute(np.fft.fft(a_matrix, 256))

                neighbors = [target_id-1, target_id,target_id + 1]
                if target_id == 0:
                    neighbors.remove(-1)
                elif target_id == 255:
                    neighbors.remove(256)
                raw_mag_frames = mag_frames[raw, :]
                neighbors_wave = [raw_mag_frames[neighbor] for neighbor in neighbors]
                max_wave = max(neighbors_wave)
                max_wave_list.append(max_wave)
                if max_wave > wave_star[star_id]:
                    wave_star[star_id] = max_wave
                    id_adjust[star_id] = neighbors[neighbors_wave.index(max_wave)] # 如果产生了某个id的更大值，进行修正
                    angle_star[star_id] = angle
                    a_matrix_star[star_id] = a_matrix
            plt.plot(angle_grid,max_wave_list)
            plt.savefig('phi_wave.png')
            plt.show()
        mag_frames = [np.absolute(np.fft.fft(i, 256)) for i in a_matrix_star ]
        return np.asarray(angle_star),id_adjust,mag_frames


    def adjust6(self,X,threshold = 20):
        c = self.c
        gamma = self.gamma
        L = self.L
        T_s = self.T_s
        N_a = self.N_a
        d = L / (N_a - 1)
        f0 = self.f0

        angle_star, id_adjust, mag_frames = self.adjust_angle3(X, threshold=threshold)
        stat4 = self.FFT_for_distance_id(X, threshold=threshold)
        id_combination = list(stat4[0][0]) # 原来的ID
        for i in range(len(id_combination)):
            # if id_combination[i] > 128:
            #     id_combination[i] = id_combination[i] - 256
            id_adjust[i] = c/2/gamma/T_s * (id_combination[i]/256 - angle_star[i]/2/np.pi)
        # for i in range(len(id_combination)):
        #     comang = c / f0 / d / 2 * id_combination[i] / 86 - angle_star[i] * c / f0 / d / 4 / np.pi
        #     if comang > 1:
        #         comang = comang-2
        #     id_adjust[i] = -np.arcsin(comang) / np.pi * 180
        return id_adjust


if __name__ == '__main__':
    sample_tool = Sample()
    FFT_tool = FFT()
    objects = np.asarray([(8, 0, 5),(5.5, -np.pi/4, 3),(1.42, np.pi/6, 2),(6.05, np.pi/5, 5)])
    # objects = np.asarray([(4, 0, 5)])
    # objects = np.asarray([(6,0,1)])
    X = sample_tool.sample(objects)
    X = np.load('data_q2.npy')
    noise = 'big'


    if noise == 'big': # 如果有明显白噪音的干扰
        distance_threshold = 200 # 用于距离找峰值的阈值，滤掉噪音
        angle_threshold = 100 # 用于角度找峰值的阈值，滤掉噪音
    else:
        distance_threshold = 50  # 用于距离找峰值的阈值，滤掉噪音
        angle_threshold = 20  # 用于角度找峰值的阈值，滤掉噪音



    d_mat = FFT_tool.FFT_for_distance(X)
    stat = FFT_tool.FFT_for_distance_star(X,threshold=distance_threshold)
    distance_star = list(stat[0][0])
    print('distance:',distance_star)

    theta_mat = FFT_tool.FFT_for_angle(X)
    stat = FFT_tool.FFT_for_angle_star(X,threshold=angle_threshold)
    angle_star = list(stat[0][0])
    print('angle:',angle_star)

    stat4 = FFT_tool.FFT_for_angle_id(X, threshold=angle_threshold)
    print(stat4)
    id_combination = list(stat4[0][0])  # 原来的ID
    print(id_combination)
    # raise Exception


    angle_star, id_adjust, mag_frames = FFT_tool.adjust_angle2(X,threshold=angle_threshold)
    print('adjust angle:', FFT_tool.adjust5(X,threshold=angle_threshold))

    print('adjust dis:', FFT_tool.adjust6(X, threshold=distance_threshold))


