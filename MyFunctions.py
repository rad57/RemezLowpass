import numpy as np

# 理想LPF
# 通過域端, 阻止域端, 求めたい周波数
def IdealLowpass(Fp, Fs, f):
    # fが0以上でFp以下なら通過域 (振幅特性は1.0)
    if (f >= 0) & (f <= Fp):
        return 1.0
    # それ以外は0.0
    else:
        return 0.0
# ========== IdealLowpass() ==========

# 係数行列Aを求める (数がM+2個に整形済みの参照点を渡すこと)
# 極値周波数の配列
def CalcAMatrix(ExtremePoints):

    # 係数行列用の配列を作成
    size = len(ExtremePoints)
    A = np.zeros((size, size))

    for i in range(size):
        f = ExtremePoints[i]  # 参照する周波数
        for j in range(size):
            A[i][j] = np.cos(j * 2*np.pi*f)

    # 最後の列は+1と-1が交互に配置される(交番定理)
    for i in range(size):
        A[i][size - 1] = (-1)**i

    return A
# ========== CalcAMatrix ==========

# 指定箇所での振幅を求める
# 連立方程式を解いたあとの配列 求めたい場所の周波数
def GetAmpitude(x, f):
    a = x[0:(len(x)-1)]
    w = 2*np.pi*f
    sum = 0.0

    for i in range(len(a)):
        sum += a[i]*np.cos(w*i)

    return sum

# ========== GetAmplitude ==========

# 振幅特性を求める
# 連立方程式を解いたあとの配列(最後の要素は最大誤差であることに注意!!) 周波数分割数
def CalcAmplitudeResponce(x, S):
    # 最後の要素は不要なのでカットしてフィルタ係数を求める
    a = x[0:(len(x)-1)]

    res = np.zeros(S)

    for i in range(S):
        f = i/(2.0 * S)
        w = 2*np.pi*f
        sum = 0.0

        for j in range(len(a)):
            sum += a[j]*np.cos(w*j)
    
        res[i] = sum

    return res
# ========== CalcAmplitudeResponce ==========

# 振幅特性から極値を求めて配列に列挙する
def GetNextExtremePoints(x, S):
    # 振幅特性を求める
    res = CalcAmplitudeResponce(x, S)

    # 極値周波数リスト
    peaks = np.zeros(0)

    for i in range(1, S-1):
        # iとその両サイドのインデックス
        front = i - 1
        back = i + 1

        # 参照点iでの振幅とその前後で比較する
        dFront = res[i] - res[front]
        dBack = res[i] - res[back]

        # 両方の差分の符号が一致している(掛けたら正の数になる)ならその点iはピーク
        if dFront*dBack > 0:
            f = i / (2.0 * S)
            peaks = np.append(peaks, f)
        
    return peaks
# ========== GetNextExtremePoints ==========

# 極値周波数のリストの整形
def ReshapePeaks(newPeaks, M, Fp, Fs, x):
    # 整形前の極値周波数の数
    m = len(newPeaks)

    # 極値周波数の数がM+2より多いなら誤差の大きい場所から消していく
    if m > M+2:
        #数が一致するまで...
        while(m != M+2):
            pLen = np.zeros(0)
            # 1つ後の極値周波数との距離を求める
            for i in range(len(newPeaks)-1):
                l = 0
                # 固定されている位置については距離はでかい数字
                if newPeaks[i] == 0.0:
                    l = 999
                elif newPeaks[i] == Fp:
                    l = 999
                elif newPeaks[i] == Fs:
                    l = 999
                elif newPeaks[i] == 0.5:  # 0.5については踏まないのでは?
                    l = 999
                else:
                    l = newPeaks[i+1] - newPeaks[i]

                pLen = np.append(pLen, l)

            # 隣との距離が一番近いやつを消す
            index = np.argmin(pLen)
            newPeaks = np.delete(newPeaks, index)

            # 極値周波数の数mの更新
            m = len(newPeaks)
    elif m < M+2:
        print("oh no")
        
    return newPeaks
# ========== ReshapePeaks ==========
