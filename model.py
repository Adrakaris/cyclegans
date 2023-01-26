from collections import deque
from datetime import datetime
from mimetypes import init
import os
from enum import Enum
import random
import numpy as np
import matplotlib.pyplot as plt 
import pickle as pkl

import tensorflow as tf 
keras = tf.keras

from loader import Loader, Domain, Sampler
from components import Norm, convLeakyRelu, convBlock, convTransposeBlock, convUpsampleUnet, resBlock
from keras.initializers import RandomNormal  # type:ignore
from keras.layers import Input, UpSampling2D, Concatenate
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model

class GenType(Enum):
    UNET = "unet"
    RESNET = "resnet"
    HCDENS = "dens"


class CycleGan:
    def __init__(self,
                 inputDim=(128,128,3),
                 learningRate=0.0002,
                 lambdaValidation=1,
                 lambdaReconstr=10,
                 lambdaIdentity=1,
                 genType=GenType.UNET,
                 epoch=0) -> None:
        self.inputDim = inputDim
        self.lr = learningRate
        self.beta1 = 0.5
        self.lamValid = lambdaValidation
        self.lamRec = lambdaReconstr
        self.lamId = lambdaIdentity
        self.genType = genType
        self.bufferMaxLength = 50
        self.channels = inputDim[2]
        
        self.epoch = epoch
        
        self.buffer_A = deque(maxlen=self.bufferMaxLength)
        self.buffer_B = deque(maxlen=self.bufferMaxLength)
        
        # calculate output shape of patchGAN D
        self.discPatch = (8, 8, 1)
        
        # weight initialiser
        self.winit = RandomNormal(seed=42, mean=0, stddev=0.02)
        
        self.compileModels()
    
    
    def trainDiscriminators(self, inAs, inBs, ONES, ZEROS):
        # fake
        fakeB = self.gAB(inAs)
        fakeA = self.gBA(inBs)
        
        self.buffer_A.append(fakeA)
        self.buffer_B.append(fakeB)
        
        fakeArand = random.sample(self.buffer_A, min(len(self.buffer_A), len(inAs)))
        fakeBrand = random.sample(self.buffer_B, min(len(self.buffer_B), len(inBs)))     
        
        daLoss1 = self.dA.train_on_batch(inAs, ONES)
        daLoss0 = self.dA.train_on_batch(fakeArand, ZEROS)
        daLoss = 0.5 * np.add(daLoss1, daLoss0)
        
        dBLoss1 = self.dB.train_on_batch(inBs, ONES)
        dBLoss0 = self.dB.train_on_batch(fakeBrand, ZEROS)
        dbLoss = 0.5 * np.add(dBLoss1, dBLoss0)
        
        return 0.5 * np.add(daLoss, dbLoss)
    
    def trainGenerators(self, inAs, inBs, ONES, ZEROS):
        return self.combined.train_on_batch(
            [inAs, inBs],
            [ONES, ONES, inAs, inBs, inAs, inBs]
        )
        
    def train(self, dataLoader:Loader, runFolder, epochs, testAid, testBid, batchSize=1, sampleInverval=150, epochSaveInterval=10, skipDiscriminator=False):
        starttime = datetime.now()
        
        # adversarial loss ground truth
        ONES = np.ones((batchSize,) + self.discPatch)
        ZEROS = np.zeros((batchSize,) + self.discPatch)
        
        epoch = -1
        try:
            for epoch in range(self.epoch, epochs):
                for batch_i, (imA, imB) in enumerate(dataLoader.load_batch()):
                    
                    if not skipDiscriminator:
                        dLoss = self.trainDiscriminators(imA, imB, ONES, ZEROS)
                    else:
                        la = self.dA.evaluate(self.gBA(imA), ZEROS)
                        lb = self.dA.evaluate(self.gAB(imB), ZEROS)
                        dLoss = 0.5 * np.add(la, lb)
                    gLoss = self.trainGenerators(imA, imB, ONES, ZEROS) 
                    
                    now = datetime.now() - starttime
                    
                    print(f"[Epoch {epoch}/{epochs}] \
[Batch {batch_i}] [D loss {dLoss[0]:.5f} / Acc {dLoss[1]*100:.1f}%] \
[G loss {gLoss[0]:.5f} adv {np.sum(gLoss[1:3]):.3f} cyc {np.sum(gLoss[3:5]):.3f} id {np.sum(gLoss[5:7]):.3f}] \
[Time {now}]", end="\r")
                    
                    if batch_i % sampleInverval == 0:
                        self.sampleImages(dataLoader, runFolder, batch_i, epoch, testAid, testBid)
                print()
                if epoch % epochSaveInterval == 0:
                    self.combined.save_weights(os.path.join(runFolder, f"weights/weights-{epoch}.h5"))
                    self.combined.save_weights(os.path.join(runFolder, "weights/weights.h5"))
                    self.saveModel(runFolder)
                
                    
        except KeyboardInterrupt:
            print("Finishing early")
            self.combined.save_weights(os.path.join(runFolder, f"weights/weights-{epoch}.h5"))
            self.combined.save_weights(os.path.join(runFolder, "weights/weights.h5"))
            self.saveModel(runFolder)
    
    
        
    def compileModels(self):
        self.dA = self.discriminator("disc-A")
        self.dB = self.discriminator("disc-B")
        
        self.dA.compile(loss="mse", optimizer=Adam(self.lr, self.beta1), metrics=["accuracy"])
        self.dB.compile(loss="mse", optimizer=Adam(self.lr, self.beta1), metrics=["accuracy"])
        
        self.gAB: Model
        self.gBA: Model 
        if self.genType == GenType.UNET:
            self.gAB = self.generatorUnet("unet_g_A_to_B")
            self.gBA = self.generatorUnet("unet_g_B_to_A")
        elif self.genType == GenType.RESNET:
            self.gAB = self.generatorResnet("resnet_g_A_to_B")
            self.gBA = self.generatorResnet("resnet_g_B_to_A")
        elif self.genType == GenType.HCDENS:
            self.gAB = self.generatorDensenet("densnet_g_A_to_B")
            self.gBA = self.generatorDensenet("densnet_g_B_to_A")
        
        self.dA.trainable = False
        self.dB.trainable = False 
        
        # input images
        inA = Input(self.inputDim)
        inB = Input(self.inputDim)
        # translated
        fakeB = self.gAB(inA)
        fakeA = self.gBA(inB)
        # reconstructed
        recA = self.gBA(fakeB)
        recB = self.gAB(fakeA)
        # identity
        idA = self.gBA(inA)
        idB = self.gAB(inB)
        # adversarial loss
        validA = self.dA(fakeA)
        validB = self.dB(fakeB)
        # combined
        self.combined = Model(
            inputs=[inA, inB],
            outputs=[validA, validB, recA, recB, idA, idB]
        )
        self.combined.compile(
            loss=["mse", "mse", "mae", "mae", "mae", "mae"],
            loss_weights=[self.lamValid, self.lamValid, self.lamRec, self.lamRec, self.lamId, self.lamId],
            optimizer=Adam(self.lr, self.beta1)
        )
        
        self.dA.trainable=True
        self.dB.trainable=True
    
    def discriminator(self, name):
        img = Input(shape=self.inputDim)
        # 128 128 (3)
        
        y1 = convLeakyRelu(img, 32, norm=Norm.NONE, initialiser=self.winit)
        # 64 64 32
        y2 = convLeakyRelu(y1, 64, initialiser=self.winit)
        # 32 32 64
        y3 = convLeakyRelu(y2, 128, initialiser=self.winit)
        # 16 16 256
        y4 = convLeakyRelu(y3, 256, initialiser=self.winit)
        
        output = Conv2D(filters=1, kernel_size=4, strides=1, padding="same", 
                        kernel_initializer=self.winit)(y4)
        # 16 16 1
        return Model(img, output, name=name)
    
    def generatorUnet(self, name):
        img = Input(shape=self.inputDim)
        
        # downsampling
        d1 = convBlock(img, 32, 4, initialiser=self.winit)
        d2 = convBlock(d1, 64, 4, initialiser=self.winit)
        d3 = convBlock(d2, 128, 4, initialiser=self.winit)
        d4 = convBlock(d3, 256, 4, initialiser=self.winit)
        
        # upsampling
        u1 = convUpsampleUnet(d4, d3, 128, 4, initialiser=self.winit)
        u2 = convUpsampleUnet(u1, d2, 64, 4, initialiser=self.winit)
        u3 = convUpsampleUnet(u2, d1, 32, 4, initialiser=self.winit)
        u4 = UpSampling2D(size=2)(u3)
        out = Conv2D(self.channels, kernel_size=4, strides=1, padding="same", 
                     kernel_initializer=self.winit, activation="tanh")(u4)
        
        return Model(img, out, name=name)

    def generatorResnet(self, name):
        img = Input(shape=self.inputDim)
        
        # downsampling
        d1 = convBlock(img, 32, 7, 1, initialiser=self.winit)
        d2 = convBlock(d1, 64, 3, 2, initialiser=self.winit)
        d3 = convBlock(d2, 128, 3, 2, initialiser=self.winit)
        
        # resnet
        r = d3
        for _ in range(9):
            r = resBlock(r, 128, 3, initialiser=self.winit)
        
        # upsampling
        u1 = convTransposeBlock(r, 64, 3, initialiser=self.winit)
        u2 = convTransposeBlock(u1, 32, 3, initialiser=self.winit)
        out = Conv2D(filters=3, kernel_size=7, strides=1, padding="same", kernel_initializer=self.winit, activation="tanh")(u2)
        
        return Model(img, out, name=name)

    def generatorDensenet(self, name):
        img = Input(shape=self.inputDim)
        
        # downsampling
        e1 = convBlock(img, 32, 7, 1, initialiser=self.winit)
        e2 = convBlock(e1, 64, 3, 2, initialiser=self.winit)
        e3 = convBlock(e2, 128, 3, 2, initialiser=self.winit)
        
        # dense block
        d1 = convBlock(e3, 256, 3, 1, initialiser=self.winit)
        d2 = convBlock(d1, 256, 3, 1, initialiser=self.winit)
        d2conc = Concatenate()([d1, d2])
        d3 = convBlock(d2conc, 256, 3, 1, initialiser=self.winit)
        d3conc = Concatenate()([d1,d2,d3])
        d4 = convBlock(d3conc, 256, 3, 1, initialiser=self.winit)
        d4conc = Concatenate()([d1, d2, d3, d4])
        d5 = convBlock(d4conc, 128, 3, 1, initialiser=self.winit)
        
        # upsampling
        u1 = convTransposeBlock(d5, 64, 3, initialiser=self.winit)
        u2 = convTransposeBlock(u1, 32, 3, initialiser=self.winit)
        out = Conv2D(filters=3, kernel_size=7, strides=1, padding="same", kernel_initializer=self.winit, activation="tanh")(u2)
        
        return Model(img, out, name=name)


    def sampleImages(self, dataLoader:Loader, runFolder, batchNo, epoch, aIndex, bIndex):
        row,coln = 2,4
        
        # display the constant and randomly picked images
        for p in range(2):
            if p == 1:
                ia = dataLoader.load_data(Domain.X, isTesting=True)
                ib = dataLoader.load_data(Domain.Y, isTesting=True)
            else:
                ia = dataLoader.load_one(Domain.X, aIndex)
                ib = dataLoader.load_one(Domain.Y, bIndex)
                
            if len(ia.shape) == 3:
                ia = np.expand_dims(ia, axis=0)
                ib = np.expand_dims(ib, axis=0)
            
            fakeb = self.gAB(ia)
            fakea = self.gBA(ib)
            
            reca = self.gBA(fakeb)
            recb = self.gAB(fakea)
            
            ida = self.gBA(ia)
            idb = self.gAB(ib)
            
            genImages = np.concatenate([ia, fakeb, reca, ida, ib, fakea, recb, idb])  # type:ignore
            
            # rescale
            genImages = 0.5 * genImages + 0.5  # type:ignore
            genImages = np.clip(genImages, 0, 1)
            
            # plots
            titles = ["original", "translated", "reconstructed", "identity"]
            fig, axis = plt.subplots(row, coln, figsize=(12, 6))
            count = 0
            for i in range(row):
                for j in range(coln):
                    axis[i,j].imshow(genImages[count])  # type:ignore
                    axis[i,j].set_title(titles[j])  # type:ignore
                    axis[i,j].axis("off")  # type:ignore
                    count += 1
            fig.savefig(os.path.join(runFolder, f"images/{p}_{epoch}_{batchNo}.png"))
            plt.close()
        
    def predict(self, generator:Sampler, characters:str, res:int=128, direction="a"):
        """direction can be a or b"""
        images = generator.generateImages(characters, res)
        print(f"Images shape: {images.shape}")
        if direction == "a":
            outChars = self.gAB.predict(images)
        else:
            outChars = self.gBA.predict(images)
        print(f"Output shape: {outChars.shape}")
        
        outChars = 0.5 * outChars + 0.5
        outChars = np.clip(outChars, 0, 1)
        
        return outChars
            
    def plotModels(self, runPath):
        plot_model(self.gAB, runPath + os.sep + "vis/gen_ab.png", show_shapes=True)
        plot_model(self.dA, runPath + os.sep + "vis/dis_a.png", show_shapes=True)
        plot_model(self.combined, runPath + os.sep + "vis/combined.png", show_shapes=True)

    def saveModel(self, runFolder):
        self.combined.save(os.path.join(runFolder, "model.h5"))
        self.dA.save(os.path.join(runFolder, 'd_A.h5') )
        self.dB.save(os.path.join(runFolder, 'd_B.h5') )
        self.gBA.save(os.path.join(runFolder, 'g_BA.h5')  )
        self.gAB.save(os.path.join(runFolder, 'g_AB.h5') )  
        
        pkl.dump(self, open(os.path.join(runFolder, "obj,pk1"), "wb"))
        
    def loadWeights(self, path:str):
        self.combined.load_weights(os.path.join(path, "model.h5"))
        self.dA.load_weights(os.path.join(path, "d_A.h5"))
        self.dB.load_weights(os.path.join(path, "d_B.h5"))
        self.gBA.load_weights(os.path.join(path, "g_BA.h5"))
        self.gAB.load_weights(os.path.join(path, "g_AB.h5"))

    def loadCombinedFromCheckpoint(self, path:str):
        self.combined.load_weights(path)
