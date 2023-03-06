from enum import Enum
from glob import glob
import math
from typing import Tuple
from PIL import ImageFont, Image, ImageDraw

import cv2
import numpy as np
import os
import random
import sys 

from sklearn.model_selection import train_test_split 


PRESETCHARACTERS = "天地玄黄宇宙洪荒日月盈昃辰宿列张\
寒来暑往秋收冬藏闰余成岁律吕调阳\
云腾致雨露结为霜金生丽水玉出昆冈\
剑号巨阙珠称夜光果珍李柰菜重芥姜\
海咸河淡鳞潜羽翔龙师火帝鸟官人皇\
始制文字乃服衣裳推位让国有虞陶唐\
吊民伐罪周发殷汤坐朝问道垂拱平章\
爱育黎首臣伏戎羌遐迩一体率宾归王\
鸣凤在竹白驹食场化被草木赖及万方\
盖此身发四大五常恭惟鞠养岂敢毁伤\
女慕贞洁男效才良知过必改得能莫忘\
罔谈彼短靡恃己长信使可覆器欲难量\
墨悲丝染诗赞羔羊景行维贤克念作圣\
德建名立形端表正空谷传声虚堂习听\
祸因恶积福缘善庆尺璧非宝寸阴是竞\
资父事君曰严与敬孝当竭力忠则尽命\
临深履薄夙兴温凊似兰斯馨如松之盛\
川流不息渊澄取映容止若思言辞安定\
笃初诚美慎终宜令荣业所基籍甚无竟\
学优登仕摄职从政存以甘棠去而益咏\
乐殊贵贱礼别尊卑上和下睦夫唱妇随\
外受傅训入奉母仪诸姑伯叔犹子比儿\
孔怀兄弟同气连枝交友投分切磨箴规\
仁慈隐恻造次弗离节义廉退颠沛匪亏"


class Domain(Enum):
    X = 0
    Y = 1


def generateImages(fontpath:str, outpath:str, res=128, characters=PRESETCHARACTERS) -> None:
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        
    font = ImageFont.truetype(fontpath, size=res-30)
    WIDTH = res-30
    HEIGHT = int(WIDTH*1)
    
    for i, char in enumerate(characters):
        img = Image.new("RGB", (res, res), color="white")
        
        draw = ImageDraw.Draw(img)
        
        draw.text(((res-WIDTH)/2, (res-HEIGHT)/2), char, font=font, fill=0, align="center")
        
        img.save(os.path.join(outpath, f"{i}.png"), "png")


if __name__ == "__main__":
    args = sys.argv
    print(args)
    if len(args) == 3:
        generateImages(args[1], args[2])
    elif len(args) == 4:
        generateImages(args[1], args[2], int(args[3]))
    elif len(args) == 5:
        generateImages(args[1], args[2], int(args[3]), args[4])
    else:
        print("Args: <font path> <out folder> [resolution=128] [characters]")


class Loader:
    def __init__(self, dataA, dataB, res=(128,128)) -> None:
        self.dataA = dataA
        self.dataB = dataB
        self.res = res
        self.nBatch = -1
    
    def load_data(self, domain:Domain, batchSize=1, isTesting=False):
        raise NotImplementedError()
    
    def load_batch(self, batchSize=1, isTesting=False):
        raise NotImplementedError()
    
    def load_one(self, domain:Domain, index:int) -> np.ndarray:
        raise NotImplementedError()
    
    def imread(self, path, imgType):
        i = cv2.imread(path, imgType)
        # i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        i = np.divide(i, 127.5) - 1.0  # standardise
        return i.astype(np.float32)
    
    
class Sampler:
    def __init__(self, fontURL) -> None:
        self.fontURL = fontURL
        
    def generateImages(self, characters:str, res=128, shrink=30):
            
        font = ImageFont.truetype(self.fontURL, size=res-shrink)  # was 30
        WIDTH = res-shrink
        HEIGHT = int(WIDTH*1)
        
        # total image width
        # imRows = math.ceil(len(characters) - colWid) 
        out = []
        for i, char in enumerate(characters):
            img = Image.new("RGB", (res, res), color="white")
            
            draw = ImageDraw.Draw(img)
            
            draw.text(((res-WIDTH)/2, (res-HEIGHT)/2), char, font=font, fill=0, align="center")
            
            out.append(np.array(img))
        return np.array(out)
            
    def stitchImages(self, images, characterRes=128, columns=16):
        # # work out full image width and height
        imRows = math.ceil(images.shape[0] / columns)
        
        BLANK = np.ones((128,128,3))
        
        rows = []
        for i in range(imRows):
            # print(f"row {i*columns} col {i*columns+columns}")
            row = list(images[i*columns:i*columns+columns])
            # print(f"len row {len(row)} row0 {row[0].shape}")
            while len(row) != columns:
                row.append(BLANK)
            # print(f"len row {len(row)}")
            row = cv2.hconcat(np.array(row))
            # print(f"row shape {row.shape}")
            # row = cv2.resize(row, )
            # img = Image.new("RGB", (characterRes*columns, characterRes))
            # img.paste(row, (0,0,characterRes,len(row)*characterRes))
            # img.paste(row)
            # rows.append(img)
            rows.append(row)
        block = cv2.vconcat(np.array(rows))  
        
        return block 
    
    def stitchImagesVert(self, images, characterRes=128, columns=16):
        # # work out full image width and height
        imRows = math.ceil(images.shape[0] / columns)
        
        BLANK = np.ones((128,128,3))
        
        rows = []
        for i in range(imRows):
            # print(f"row {i*columns} col {i*columns+columns}")
            row = list(images[i*columns:i*columns+columns])
            # print(f"len row {len(row)} row0 {row[0].shape}")
            while len(row) != columns:
                row.append(BLANK)
            # print(f"len row {len(row)}")
            row = cv2.vconcat(np.array(row))
            # print(f"row shape {row.shape}")
            # row = cv2.resize(row, )
            # img = Image.new("RGB", (characterRes*columns, characterRes))
            # img.paste(row, (0,0,characterRes,len(row)*characterRes))
            # img.paste(row)
            # rows.append(img)
            rows.append(row)
        block = cv2.hconcat(np.array(list(reversed(rows))))  
        
        return block 
    
    
class KindaLoadEverything(Loader):
    def __init__(self, dataA, dataB, res=(128,128)) -> None:
        super().__init__(dataA, dataB, res)
        
        self.imgX = []
        self.imgY = []
        
        self.loadAll()
        
        # self.xTrain, self.xTest = train_test_split(self.imgX, test_size=0.2)
        # self.yTrain, self.yTest = train_test_split(self.imgY, test_size=0.2)
        TRAINSIZE = 0.8
        propX = int(len(self.imgX) * TRAINSIZE)+1
        propY = int(len(self.imgY) * TRAINSIZE)+1
        
        np.random.shuffle(self.imgX)
        np.random.shuffle(self.imgY)
        
        self.xTrain = self.imgX[:propX]
        self.xTest = self.imgX[propX:]
        self.yTrain = self.imgY[:propY]
        self.yTest = self.imgY[propY:]
        
        
    def loadAll(self):
        pathA = glob(self.dataA + "/*")
        pathB = glob(self.dataB + "/*")
        
        # load everyhting into memory; there's not much there anyway
        for x in pathA:
            img = self.imread(x, cv2.IMREAD_ANYCOLOR)
            img = cv2.resize(img, self.res, interpolation=cv2.INTER_CUBIC)
            self.imgX.append(img)
        
        for y in pathB:
            # pathB contains lanting which is greyscale
            imy = self.imread(y, cv2.IMREAD_GRAYSCALE)
            imy = cv2.resize(imy, self.res, interpolation=cv2.INTER_CUBIC)
            imgRGB = np.repeat(imy[:, :, np.newaxis], 3, axis=2)
            self.imgY.append(imgRGB)
            
    def load_one(self, domain:Domain, index:int):
        if domain == Domain.X:
            return self.imgX[index]
        elif domain == Domain.Y:
            return self.imgY[index]
        
    def load_batch(self, batchSize=1, isTesting=False):
        if isTesting:
            xData = self.xTest
            yData = self.yTest
        else:
            xData = self.xTrain 
            yData = self.yTrain
        
        # print(xData, yData)
        
        self.nBatch = int(min(len(xData), len(yData)) / batchSize)
        samples = self.nBatch * batchSize
        
        # samplesX = np.random.choice(xData, samples, replace=False)
        # samplesY = np.random.choice(yData, samples, replace=False)
        samplesX = random.sample(xData, k=samples)
        samplesY = random.sample(yData, k=samples)
        
        for i in range(self.nBatch-1):
            batchX = samplesX[i*batchSize:(i+1)*batchSize]
            batchY = samplesY[i*batchSize:(i+1)*batchSize]
            yield np.array(batchX), np.array(batchY)
            
    def load_data(self, domain: Domain, batchSize=1, isTesting=False):
        if isTesting:
            xData = self.xTest
            yData = self.yTest
        else:
            xData = self.xTrain 
            yData = self.yTrain
            
        self.nBatch = int(min(len(xData), len(yData)) / batchSize)
        samples = self.nBatch * batchSize
        
        if domain == Domain.X:
            samplesX = random.sample(xData, k=samples)
            i = np.random.randint(0, self.nBatch-1)
            batchX = samplesX[i*batchSize:(i+1)*batchSize]
            return np.array(batchX)
        elif domain == Domain.Y:
            samplesY = random.sample(yData, k=samples)
            i = np.random.randint(0, self.nBatch-1)
            batchY = samplesY[i*batchSize:(i+1)*batchSize]
            return np.array(batchY)
        
    
class KindaLoadEverythingGrey(KindaLoadEverything):
    def __init__(self, dataA, dataB, res=(128, 128)) -> None:
        super().__init__(dataA, dataB, res)
        
    def loadAll(self):
        pathA = glob(self.dataA + "/*")
        pathB = glob(self.dataB + "/*")
        
        # load everyhting into memory; there's not much there anyway
        for x in pathA:
            img = self.imread(x, cv2.IMREAD_ANYCOLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, self.res, interpolation=cv2.INTER_CUBIC)
            self.imgX.append(img[..., np.newaxis])
        
        for y in pathB:
            # pathB contains lanting which is greyscale
            imy = self.imread(y, cv2.IMREAD_GRAYSCALE)
            imy = cv2.resize(imy, self.res, interpolation=cv2.INTER_CUBIC)
            # imgRGB = np.repeat(imy[:, :, np.newaxis], 3, axis=2)
            self.imgY.append(imy[..., np.newaxis])

