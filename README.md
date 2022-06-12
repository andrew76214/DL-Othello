# DL-Othello
## 範例程式觀察
從OthelloModel.py裡可以觀察到，範例程式的骨幹網路(backbone)總共有五層，而backbone裡的layer都是使用Conv2D且每一層都會視情況進行正規化(BatchNormalization)。
第一層使用兩個filter大小為128的layer，之後再加入relu活化函數，使得訓練可以神經網路可以更加活化學習，避免像是線性函數一樣較為死板。
第二、三層使用三層及二層filter大小為256的layer，且每層結束時都會加入relu活化函數進行活化，但有趣的是第二層比第三層多一層filter大小為256的layer，但我不太清楚多加入這層的用意。
第四、五層使用三層及二層filter大小為512的layer，與第二、三層極為相似，每一層結束時也會加入relu活化函數進行活化。

## 嘗試與調整
### 第一版：
我減少了backbone的層數，使其來到了四層，但在每一層裡都加入三至四層與範例程式一樣的layer，在每一層結束時加入relu活化函數，與範例程式不一樣的是，我在前二層結束時都加入了max pooling，在後兩層結束時則是加入average pooling，num_filter的部分則是前面兩層都是128，後兩層則是256，我有嘗試過大小為32、64、512的num_filter，但是效果卻不盡人意。
在FC層(Fully connected layer)中，讓模型flatten之後就直接Dense輸出了，並沒有加入dropout或是GlobalAveragePooling2D等等。
			
### 第二版：     
基本上已經與範例程式的模型架構大相逕庭了，在backbone中，我先加入Conv2D、BatchNormalization、relu各一層，之後再加入五層num_filter大小為128的layer，而每層layer裡，加入了Conv2D、BatchNormalization、relu各二層，之後在backbone中，再加入Conv2D、BatchNormalization、relu各一層。
在FC層中，與第一版很像，讓模型flatten之後就直接Dense輸出了，並沒有加入dropout或是GlobalAveragePooling2D等等。

## 測試模型勝率：
 - 第一版 -> 60%
 - 第二版 -> 40%

## 心得：
這次在網路上搜尋這次專題相關的資料和文獻時，我有參考這篇論文
[P. Liskowski, W. Jaśkowski and K. Krawiec, "Learning to Play Othello With Deep Neural Networks," in IEEE Transactions on Games, vol. 10, no. 4, pp. 354-364, Dec. 2018, doi: 10.1109/TG.2018.2799997.](https://arxiv.org/pdf/1711.06583.pdf)，
但是時間不太夠且顯卡只有6GB VRAM而已，沒辦法嘗試更深的模型，實屬遺憾。
	在訓練的時候，我本來以為loss越低越好，但是事實上好像並非如此，第一版模型的最終loss大概在0.46左右，第二版模型的最終loss大概在0.18左右，但是實際跟機器人pk的時候，勝率卻是五五開，有時更糟甚至是六四開；還是其實loss和勝率呈負相關，而我的模型的loss還太高，樣本數太少無法得出結論。

## 訓練環境：
CPU: 		Intel I5-10400

GPU: 		Nivdia GTX 1660

System: 	Windows 11

tensorflow_gpu-2.8.0

python 3.10

cuDNN 8.1

CUDA 11.2
