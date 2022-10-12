from fft2d import * # 引用刚才定义的二维FFT算法，解决题目四个问题
import numpy as np

def solve_ff2d(X,threshold):
    ff2 = FFT2d(X)
    top_id = find_wave(ff2, threshold)
    count = 1
    for id_combination in top_id:
        theta_id, R_id = id_combination
        print(f'半径{count}:',id2R(R_id))
        print(f'角度{count}:',id2theta(theta_id))
        print('\n')
        count += 1

# q1
print('q1')
X = np.load('data_q1.npy')
solve_ff2d(X,threshold=0)

# q2
print('q2')
X = np.load('data_q2.npy')
solve_ff2d(X,threshold=1e4) # 设置阈值的作用是将比较小的峰值过滤掉，尤其在存在白噪音时，要将阈值调大

#q3
print('q3')
frame_th = 1
for X in np.load('data_q3.npy'):
    print(f'{frame_th} frame')
    solve_ff2d(X,threshold=5e3)
    frame_th += 1

#q4
print('q4')
X = np.load('data_q4.npy')
solve_ff2d(X,threshold=5e3)