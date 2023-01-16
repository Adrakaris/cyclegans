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

mode = LOAD  

# START_EPOCH = 85  # TODO
# SKIPDIS = False
# SAVEINTERVAL = 10  # 10
#endregion
