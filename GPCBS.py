# Written by Tingrui Hu, <hutingrui@mail.bnu.edu.cn>
# This is a demo code for 'Hyperspectral Band Selection Method Based on Global Partition Clustering'.
# The code is for research purposes only. All rights reserved.
import math # Version: 3.8.9
from scipy.io import loadmat # Version: 1.9.1
import numpy as np # Version: 1.24.3
from scipy.spatial.distance import squareform # Version: 1.9.1
from skimage.metrics import structural_similarity as compare_ssim # Version: 0.19.3
from math import log # Version: 3.8.9

def calSE(dataSet):
    numEntires = dataSet.shape[0] * dataSet.shape[1]
    labelCounts = {}
    for row in range(dataSet.shape[0]):
        for col in range(dataSet.shape[1]):
            currentLabel = dataSet[row][col]
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
    Entropy = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntires
        Entropy -= prob * log(prob, 2)
    return Entropy

def ssim_matrix(hyperIm):
    sizeBand = hyperIm.shape[-1]
    List = []
    for i in range(0, sizeBand):
        X = hyperIm[:, :, i]
        for j in range(i + 1, sizeBand):
            Y = hyperIm[:, :, j]
            List.append(compare_ssim(X, Y, gaussian_weight=True, win_size=3))
    Y = np.array(List)
    Mat_ssim = squareform(Y)
    return Mat_ssim

def cal_cc(im1, im2):
    im1 = np.array(im1, dtype=np.int32)
    im2 = np.array(im2, dtype=np.int32)
    mu1 = np.mean(im1)
    mu2 = np.mean(im2)
    mu = np.sum(((im1 - mu1) * (im2 - mu2)))
    sigma1 = np.sqrt(np.sum((im1 - mu1) ** 2))
    sigma2 = np.sqrt(np.sum((im2 - mu2) ** 2))
    sigma = sigma1 * sigma2
    cc = mu / sigma
    return cc

def correlation_matrix(hyperIm):
    sizeBand = hyperIm.shape[-1]
    List = []
    for i in range(0, sizeBand):
        X = hyperIm[:, :, i]
        for j in range(i + 1, sizeBand):
            Y = hyperIm[:, :, j]
            List.append(cal_cc(X, Y))
    Y = np.array(List)
    Mat_correlation = squareform(Y)
    return Mat_correlation

def Ed_matrix(hyperIm):
    sizeBand = hyperIm.shape[-1]
    List = []
    for i in range(0, sizeBand):
        X = hyperIm[:, :, i]
        X = np.array(X, dtype=np.int32)
        for j in range(i + 1, sizeBand):
            Y = hyperIm[:, :, j]
            Y = np.array(Y, dtype=np.int32)
            List.append(np.linalg.norm(X - Y))
    Y = np.array(List)
    Mat_correlation = squareform(Y)
    return Mat_correlation

def SR(S):
    L = S.shape[0]
    Y = S[np.triu_indices(L, k=1)]
    Y = sorted(Y, reverse=True)
    sc = np.mean(Y[math.floor(len(Y) * 0.05) - 1: math.floor(len(Y) * 0.1) - 1])

    alpha = []
    for i in range(0, L):
        temp = S[i, :]
        if len(temp[temp > sc]) == 0:
            alpha.append(0)
        else:
            alpha.append(np.mean(temp[temp > sc]))
    alpha = np.array(alpha)
    I = np.argsort(-alpha)

    varphi = np.zeros(L)
    varphi[I[0]] = 1

    for i in range(1, L):
        for j in range(i):
            if varphi[I[i]] < S[I[i], I[j]]:
                varphi[I[i]] = S[I[i], I[j]]

    varphi[I[0]] = min(varphi)
    theta = np.sqrt(1 - varphi ** 2)

    alpha = (alpha - min(alpha)) / (max(alpha) - min(alpha))
    theta = (theta - min(theta)) / (max(theta) - min(theta))
    eta = alpha * theta

    G = np.argsort(-eta)
    M = G[0:k]
    M = list(M)

    return M, G, eta

def bestTK(i):
    index = math.floor((density[i] + density[i + 1]) / 2)
    TKmin = float('inf')
    if density[i] + 1 == density[i + 1]:
        return density[i] + 1
    elif density[i + 1] - density[i] == 2:
        return density[i] + 1
    elif density[i + 1] - density[i] == 3:
        return density[i] + 2
    else:
        for tk in range(density[i] + 2, density[i + 1]):
            CD = 0
            for row in range(density[i], tk):
                for col in range(tk, density[i + 1] + 1):
                    CD += abs(R[row, col])
            CSk0 = 0
            for row in range(density[i], tk):
                for col in range(row, tk):
                    CSk0 += abs(R[row, col])
            CSk1 = 0
            for row in range(tk, density[i + 1] + 1):
                for col in range(row, density[i + 1] + 1):
                    CSk1 += abs(R[row, col])
            TKP = CD / (CSk0 * CSk1)
            if TKP < TKmin:
                TKmin = TKP
                index = tk
        return index

def fbestTK(i, BTK):
    index = BTK[i]
    TKmin = float('inf')
    if density[i] - density[i - 1] > 1:
        for tk in range(max(BTK[i - 1] + 2, density[i - 1] + 1), min(density[i] + 1, BTK[i + 1] - 1)):
            CD = 0
            for row in range(BTK[i - 1], tk):
                for col in range(tk, BTK[i + 1]):
                    CD += abs(R[row, col])
            CSk0 = 0
            for row in range(BTK[i - 1], tk):
                for col in range(row, tk):
                    CSk0 += abs(R[row, col])
            CSk1 = 0
            for row in range(tk, BTK[i + 1]):
                for col in range(row, BTK[i + 1]):
                    CSk1 += abs(R[row, col])
            TKP = CD / (CSk0 * CSk1)
            if TKP < TKmin:
                TKmin = TKP
                index = tk
        return index
    else:
        return index

if __name__ == '__main__':
    HSIname = 'paviaU'
    hyperIm = loadmat(HSIname + '.mat')[HSIname]
    sbn = [5, 10, 15, 20, 25, 30]

    sltBands = []
    band = hyperIm.shape[-1]


    S = ssim_matrix(hyperIm)
    R = correlation_matrix(hyperIm)
    Ed = Ed_matrix(hyperIm)

    entropy = np.zeros(band)
    for i in range(band):
        entropy[i] = calSE(hyperIm[:, :, i])

    entropy = (entropy - min(entropy)) / (max(entropy) - min(entropy))

    for k in sbn:
        sltBands = []
        if band / k >= 5:
            density, _, _ = SR(S)
            density = np.sort(density)

            TK = []
            TK.append(0)
            for i in range(len(density) - 1):
                TK.append(bestTK(i))
            TK.append(band - 1)

            BTK = TK.copy()
            for j in range(1, len(TK) - 2):
                BTK[j] = fbestTK(j, BTK)
            while BTK != TK:
                TK = BTK.copy()
                for j in range(1, len(BTK) - 2):
                    BTK[j] = fbestTK(j, BTK)
        else:
            _, G, _ = SR(S)
            TK = []
            TK = np.linspace(start=0, stop=band - 1, num=k + 1)
            TK = np.floor(TK)
            TK = TK.astype(np.int32)

            density = []
            for i in range(len(TK) - 1):
                density.append(TK[i] + np.argmax(G[TK[i]:TK[i + 1]]))

        clusterList = []
        for i in range(0, len(TK) - 1):
            for j in range(TK[i], TK[i + 1]):
                clusterList.append(i)
        clusterList.append(len(TK) - 2)

        _, _, r = SR(R)

        bandcluster = []
        while k != 0:
            f = np.zeros(band)
            n = np.zeros(band)
            for i in range(band):
                for j in density:
                    if clusterList[i] != clusterList[j]:
                        f[i] += Ed[i, j]

            f = (f - np.min(f)) / (np.max(f) - np.min(f))
            n = r * f * entropy

            indexnum = -1
            indexval = -1
            for i in range(band):
                if clusterList[i] not in bandcluster and n[i] > indexval:
                    indexval = n[i]
                    indexnum = i
            sltBands.append(indexnum)
            bandcluster.append(clusterList[indexnum])
            density[clusterList[np.argmax(n)]] = indexnum
            k -= 1

        print("Band:", sorted(sltBands))
