# 高等成像顯示技術與應用作業簡介

「高等成像顯示技術與應用」課程中完成的作業，課程聚焦於影像嵌密與加密技術的整合應用，涵蓋可逆資料隱藏、圖像隨機性測試、資訊熵分析、及密鑰敏感度評估等主題。透過彩色與灰階影像的實作任務，深入探討影像安全性評估與保護技術的實際應用。

<br>

## Assignment 01 - GWMRDH Algorithm for Color Image

將 General Weighted Modulus Reversible Data Hiding (GWMRDH) 演算法應用於彩色影像，將機密訊息嵌入後產生三張嵌密圖，再透過擷取與還原流程重建出原始圖像與訊息，並以 MSE、PSNR、嵌入容量 (EC) 與嵌入率 (ER) 等指標進行分析。

**重點**：
- 彩色圖像通道的三重嵌密架構
- 支援多種 M-ary 階進制資料隱寫
- 整合畫素值偏移處理與 RPA Table 參數操作

<br>

## Assignment 03 - Color IMEE (Integrated Message Embedding and Encryption)

本作業設計完整的圖像嵌密與加密流程：以 FCUE_C (Forward Concealment, Permutation and Encryption) 將資料嵌入彩色圖像，透過通道組合、隨機排列與矩陣轉換進行加密，對應的 BDIX_C 流程則可完整還原原圖與資料。

**重點**：
- 原圖經組合後解析度變更
- 多階段處理：GMWRDH + Channel Permutation + RT Encryption
- 適用 SIPI、Kodak 標準測試圖像

<br>

## Assignment 04 - Image Encryption Metrics (Part 1)

針對原始與加密圖像，進行以下四項安全性指標的量測分析：

1. 直方圖變異數 (Variance of Histogram)
2. 直方圖卡方檢定 (Chi-Square Test)
3. 皮爾森相關係數 (Pearson Correlation Coefficient)
4. 全域資訊熵 (Global Information Entropy)

**重點**：
- 對應圖像以彩色通道分別計算
- CSV 輸出分析各通道的加密效果

<br>

## Assignment 05 - Image Encryption Metrics (Part 2)

進一步量測影像安全性的三個指標：

5. 區域資訊熵（Local Info Entropy, LIN）
6. 圖像敏感度測試（Image Sensitivity, ISE）
7. 密鑰敏感度測試（Secret Key Sensitivity, SKS）

**重點**：
- 使用 NPCR 與 UACI 進行敏感度評估
- 支援 Stratified Sampling 的 bonus 評分
- 測試影像涵蓋不同解析度與色彩通道

<br>

## Assignment 06 - NIST Randomness Evaluation

將影像轉為二進位序列，並使用 NIST randomness testing suite 進行隨機性檢定。結果記錄 15 項隨機性指標，並整理統計為報表。

**重點**：
- Python 轉換 bit stream：`img2bin.py`
- 匯出個別測試與總結結果：NIST-10.csv
- 適用彩色與灰階圖像的隨機性評估



