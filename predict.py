import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib.pyplot as plt 

from model import CycleGan
from loader import KindaLoadEverything, Sampler

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

#region setup
BUILD = "build"
LOAD = "load"

RUN_ID = "0003"
DOM_A = "./data/a"
DOM_B = "./data/b"
RUNFOLDER = os.path.join("run", RUN_ID)  
#endregion

gan = CycleGan()

gan.loadWeights(RUNFOLDER)

sampler = Sampler("./data/simkai.ttf")
converted = gan.predict(sampler, "中化人民共和国万岁！你好中国，你好대한민국！")
