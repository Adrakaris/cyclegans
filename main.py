import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib.pyplot as plt 

from model import CycleGan
from loader import KindaLoadEverything

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

#region setup
BUILD = "build"

RUN_ID = "0001"
DOM_A = "./data/a"
DOM_B = "./data/b"
RUNFOLDER = os.path.join("run", RUN_ID)

if not os.path.exists(RUNFOLDER):
    os.makedirs(RUNFOLDER)
    os.mkdir(RUNFOLDER + "/vis")
    os.mkdir(RUNFOLDER + "/images")
    os.mkdir(RUNFOLDER + "/weights")

mode = BUILD
train = True 
#endregion

loader = KindaLoadEverything(DOM_A, DOM_B)


gan = CycleGan(inputDim=(128,128,3))

if mode == BUILD:
    gan.saveModel(RUNFOLDER)
else:
    gan.loadWeights(RUNFOLDER + "/weights")
    
gan.gAB.summary()
gan.dA.summary()
gan.combined.summary()
gan.plotModels(RUNFOLDER)


gan.train(
    dataLoader=loader,
    runFolder=RUNFOLDER,
    epochs=200,
    testAid=0,
    testBid=0,
    sampleInverval=250,
    epochSaveInterval=10
)



