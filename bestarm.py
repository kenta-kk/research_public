import numpy as np
import random
import matplotlib.pyplot as plt

# sample_means(標本平均)と、サンプリングすることを表す関数名がsampleかぶりしてて分かりにくい

# 真の平均をもとに指数分布から報酬を返すことにする
# 分散varを既知とする正規分布
var = 1     # 分散
# アームの数
K = int(10)
# アームの真の平均(固定, 一様ランダムで決めた)
true_means = [0.04799676, 0.65712573, 0.17555852, 0.56573422, 0.79480353, 0.27867847,
              0.03943761, 0.20156264, 0.64060331, 0.12827873]

def reward_sample(mean: float, variance: float ) -> float:
    reward = random.gauss(mean, variance)
    return reward

# w*の計算
# 正規分布のKL-ダイバージェンス
def kl_divergence(x: float, y: float) -> float:
    return ((x - y) ** 2) / (2 * var)

# mu1 ≠ mua
def g_a_func(x: float, mu1: float, mua: float) -> float:
    term1_1 = mu1
    term1_2 = (1 / (1 + x)) * mu1 + (x / (1 + x)) * mua
    term2_1 = mua
    term2_2 = (1 / (1 + x)) * mu1 + (x / (1 + x)) * mua
    i_1x = (1 / (1 + x)) * kl_divergence(term1_1, term1_2) + (x / (1 + x)) * kl_divergence(term2_1, term2_2)
    return (1 + x) * i_1x

def inverse_g_a_func(y:float , mu1: float, mua: float)-> float:
    return (2 * y) / ((mu1 - mua) ** 2 - 2 * y)

def f_y_func(y: float, means: list) -> float:
    sum_d_d = 0
    max_mu = max(means)
    for i in range(K):
        if means[i] != max_mu:
            term2 = (max_mu + inverse_g_a_func(y, max_mu, means[i]) * means[i]) / (1 + inverse_g_a_func(y, max_mu, means[i]))
            up_d = kl_divergence(max_mu, term2 )
            low_d = kl_divergence(means[i], term2 )
            sum_d_d += up_d / low_d
    return sum_d_d

def max_f_y(means: list)-> float:
    max_mu = max(means)
    max_2nd_mu = sorted(means)[-2]
    return kl_divergence(max_mu, max_2nd_mu)

def find_y_star(means: list, epsilon: float) -> float:
    y1 = 0
    y2 = max_f_y(means) / 2
    eq1 = f_y_func(y1, means) - 1
    eq2 = f_y_func(y2, means) - 1
    # f(x_1)f(x_2)<0 になるまで x_1とx_2 を決め直す
    if eq1 * eq2 >= 0:
        while eq1 * eq2 >= 0:
            y2 += y2 / 2
            eq2 = f_y_func(y2, means) - 1
    # 二分法のiteration
    while abs(y1 - y2) >= epsilon:
        y_middle = (y1 + y2) / 2
        if eq1 * f_y_func(y_middle, means) < 0:
            y2 = y_middle
        else:
            y1 = y_middle
    return (y1 + y2) / 2

def find_w_stars(means: list, epsilon: float) -> list:
    y_star = find_y_star(means, epsilon)
    w_stars = [0] * K
    best_arm = np.argmax(means)
    max_mean = max(means)
    x_arms = [0] * K
    x_best_arm = 1
    x_arms[best_arm] = x_best_arm
    for i in range(K):
        if i == best_arm:
            continue
        else:
            x_arms[i] = inverse_g_a_func(y=y_star, mu1=max_mean, mua=means[i])
    for i in range(K):
        w_stars[i] = x_arms[i] / sum(x_arms)
    return w_stars

# D-track
# 引数のmeansにはtrue_meansを入れる
def sample_d_track(means: list, epsilon: float):
    t = 0 # タイムステップt
    N_a_t = [0] * K # 時刻tにおけるアームaを引いた回数のリスト
    sample_means = [0] * K # 標本平均(最初はゼロ)
    # Stopping ruleは後で作って、この無限ループに取り入れる
    # とりあえずStopping ruleできるまで仮に停止時間Tを決めておく
    T = 100

    # t = 1 の時は別に切り出して書くことにする
    selected_arm_1 = random.randint(0, K-1)
    sample_reward = reward_sample(mean=means[selected_arm_1], variance=epsilon)
    t += 1
    N_a_t[selected_arm_1] += 1
    sample_means[selected_arm_1] = sample_reward

    while True:
        arms_belong_D_rule = []
        D_track_rule = np.sqrt(t) - K / 2 # 論文のU_tの右辺
        for i in range(K):
            if N_a_t[i] < D_track_rule:
                arms_belong_D_rule.append(i)
        if arms_belong_D_rule != []:
            selected_arm = np.argmin(N_a_t) # 複数あるときについて考慮できてない
            N_a_t[selected_arm] += 1
        else:
            w_stars = find_w_stars(sample_means, epsilon)
            direct_track = w_stars
            for j in range(K):
                direct_track[j] = t * w_stars[j] - N_a_t[j]
            selected_arm = np.argmax(direct_track)
            N_a_t[selected_arm] += 1
        sampled_reward = reward_sample(mean=means[selected_arm], variance=var)
        t += 1
        sample_means[selected_arm] = sample_means[selected_arm] + (sampled_reward - sample_means[selected_arm]) / N_a_t[selected_arm]
        print('N_a_t:', N_a_t)
        if t > T:
            return sample_means

def weighted_sample_mean():
def generalized_likelihood_ratio():

def stop_threshold():
