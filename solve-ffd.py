from fft2 import * # 上一个文件的名称，全部引用过来，用于解题
import numpy as np

def fft_solve(X, noise = 'big'):
    FFT_tool = FFT()

    if noise == 'big':  # 如果有明显白噪音的干扰
        distance_threshold = 200  # 用于距离找峰值的阈值，滤掉噪音
        angle_threshold = 100  # 用于角度找峰值的阈值，滤掉噪音
    else:
        distance_threshold = 50  # 用于距离找峰值的阈值，滤掉噪音
        angle_threshold = 20  # 用于角度找峰值的阈值，滤掉噪音

    d_mat = FFT_tool.FFT_for_distance(X)
    stat = FFT_tool.FFT_for_distance_star(X, threshold=distance_threshold)
    distance_star = list(stat[0][0])
    print('distance:', distance_star)

    theta_mat = FFT_tool.FFT_for_angle(X)
    stat = FFT_tool.FFT_for_angle_star(X, threshold=angle_threshold)
    angle_star = list(stat[0][0])
    print('angle:', angle_star)

    angle_star, id_adjust, mag_frames = FFT_tool.adjust_angle2(X, threshold=angle_threshold)
    angle_adjust = FFT_tool.adjust5(X, threshold=angle_threshold)
    print('adjust angle:', angle_adjust)
    print('\n')
    return distance_star, angle_star, angle_adjust

if __name__ == '__main__':
    # 1
    print('q1')
    X = np.load('data_q1.npy')
    fft_solve(X,noise='small')

    # 2
    print('q2')
    X = np.load('data_q2.npy')
    fft_solve(X, noise='big') # 第二问有白噪声的干扰

    # 3
    print('q3')
    count = 1
    for X in np.load('data_q3.npy'):
        print(f'frame{count}')
        fft_solve(X, noise='small')
        count += 1


    # 4
    print('q4')
    X = np.load('data_q4.npy')
    fft_solve(X, noise='small')

