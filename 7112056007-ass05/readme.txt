量測local information entropy(5_LIN_measure.py)：
	1.使用到套件cv2、numpy、os、csv、math、random

量測image sensitivity(6_ISE_measure.py)：
	1.使用到套件cv2、numpy、os、csv
	2.使用到的加密影像為'影像名稱_enc.png'與'影像名稱_enc_ISE.png'
	3.'影像名稱_enc_ISE.png'為原始影像中第[0,9]bit的紅色channel數值加一後再進行加密的影像

量測secret key sensitivity(7_SKS_measure.py)：
	1.使用到套件cv2、numpy、os、csv
	2.使用到的加密影像為'影像名稱_enc.png'與'影像名稱_enc_SKS.png'
	3.'影像名稱_enc_SKS.png'為Transient effect constant加1後再進行加密之影像

