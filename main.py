import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib.pyplot as plt 

from model import CycleGan
from loader import KindaLoadEverything

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

#region setup
BUILD = "build"
LOAD = "load"

RUN_ID = "0003"
DOM_A = "./data/a"
DOM_B = "./data/b"
RUNFOLDER = os.path.join("run", RUN_ID)

if not os.path.exists(RUNFOLDER):
    os.makedirs(RUNFOLDER)
    os.mkdir(RUNFOLDER + "/vis")
    os.mkdir(RUNFOLDER + "/images")
    os.mkdir(RUNFOLDER + "/weights")

mode = LOAD  # TODO
train = True 

START_EPOCH = 85  # TODO
SKIPDIS = False
SAVEINTERVAL = 10  # 10
#endregion

loader = KindaLoadEverything(DOM_A, DOM_B)


gan = CycleGan(inputDim=(128,128,3), epoch=START_EPOCH)  

if mode == BUILD:
    gan.saveModel(RUNFOLDER)
else:
    gan.loadWeights(RUNFOLDER)
    
print(f"The run number is {RUN_ID}")
if mode != BUILD or START_EPOCH != 0 or SKIPDIS:
    if mode != BUILD:
        print("GAN has been detected to be in LOAD mode")
    if START_EPOCH != 0:
        print(f"The starting epoch {START_EPOCH} is not zero")
    if SKIPDIS:
        print("The gan will be configured to not train the discriminator")
    print(f"Are you sure you want to continue? (KeyboardInterrupt to exit)")
    input()
    
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
    sampleInverval=500,
    epochSaveInterval=SAVEINTERVAL,
    skipDiscriminator=SKIPDIS
)



