import numpy as np
import matplotlib.pyplot as plt
import MyFunctions

if __name__ == "__main__":
    N = 50  # フィルタ次数
    Fp = 0.2  # 通過域端周波数
    Fs = 0.24 # 阻止域端周波数

    S = N*20  # 周波数分割数

    M = int(N/2)  # 極大/極小値の数はM+2個

    x = np.zeros(M+2)  # 設計変数
    b = np.zeros(M+2)  # 理想LPFの振幅特性
    A = np.zeros((M+2, M+2))  # 連立方程式の係数行列
    
    # ========== 初期化 ==========
    print("初期化開始")

    # 初期の参照点の数と位置を求める
    NumPassBand = int((M+2) * (Fp/(Fp+Fs)))+1
    NumStopBand = (M+2) - NumPassBand
    ExtremePoints = np.append(np.linspace(0.0, Fp, NumPassBand), np.linspace(Fs, 0.5, NumStopBand))

    # 参照点を元に理想LPFの特性を求める
    for i in range(M+2):
        b[i] = MyFunctions.IdealLowpass(Fp, Fs, ExtremePoints[i])

    # 連立方程式の係数行列を求める
    A = MyFunctions.CalcAMatrix(ExtremePoints)

    # x = A^{-1}b により連立方程式の解を求める
    # 連立方程式の求解
    invA = np.linalg.inv(A)
    x = np.dot(invA, b)

    print("初期化終了")
    # ========== 初期化 ==========

    # ========== Remezアルゴリズム本体 ==========
    # 終了判定フラグ
    isEnd = 0

    # 最大反復数
    tMax = 20

    coefsLog = np.zeros((tMax, M+2))

    for t in range(tMax):
        print("===============")
        print("t = %d" %t)
        # ピーク探索で次の極値周波数を求める
        newPeaks = MyFunctions.GetNextExtremePoints(x, S)
        
        # 両端とFp,FsをnewPeaksに含める
        newPeaks = np.append(newPeaks, 0.0)
        newPeaks = np.append(newPeaks, 0.5)
        newPeaks = np.append(newPeaks, Fp)
        newPeaks = np.append(newPeaks, Fs)

        # ソートして順番をあわせる
        newPeaks = np.sort(newPeaks)
        print("initial peaks len %d" %len(newPeaks))

        # 極値周波数の数合わせ
        newPeaks = MyFunctions.ReshapePeaks(newPeaks, M, Fp, Fs, x)

        # 参照点の変化の最大値
        e = np.max(ExtremePoints - newPeaks)

        # 新しいピークから次のフィルタ係数を求める
        ExtremePoints = newPeaks
        print("reshaped peaks len %d" %len(ExtremePoints))
        print(e)
        print(ExtremePoints)
        
        # 参照点を元に理想LPFの特性を求める
        for i in range(M+2):
            b[i] = MyFunctions.IdealLowpass(Fp, Fs, ExtremePoints[i])

        # 連立方程式の係数行列を求める
        A = MyFunctions.CalcAMatrix(ExtremePoints)

        # x = A^{-1}b により連立方程式の解を求める
        invA = np.linalg.inv(A)
        x = np.dot(invA, b)  # xの値を更新

        coefsLog[t] = x

    # end for()
    # ========== Remezアルゴリズム本体 ==========
    #plt.imshow(coefsLog.T, aspect='auto')
    #plt.show()

    res = MyFunctions.CalcAmplitudeResponce(x, S)
    f = np.linspace(0, 0.5, S)
    plt.plot(f, 20*np.log10(np.abs(res)))
    plt.show()