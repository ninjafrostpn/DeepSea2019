from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import scipy.stats as sps

plotids = ["habitatconditions"]

# Function for converting HH:MM:SS to total seconds
timetosec = np.vectorize(lambda x: (((x.hour * 60) + x.minute) * 60) + x.second)

# Extracts version number from a filename
# Filenames are of format "AllData-Z.csv", where Z is the version number
filenomatcher = re.compile(r"-[0-9]*\.")
getfileno = lambda x: int(filenomatcher.search(x)[0][1:-1])

# Read data in from the xlsx file as a pandas DataFrame (remove metadata in first 6 rows)
envdata = pd.read_excel("SOES6008_coursework_DATA.xlsx", skiprows=6)
envdata.columns = ["ActualTime", "VideoTime", "Lat", "Lon", "Dist", "Depth", "Temp", "Salinity"]
envdata = envdata.assign(VideoTimeSecs=timetosec(envdata["VideoTime"]))

if "envcorr" in plotids:
    # Look for autocorrelation in env data
    plt.subplot(2, 2, 1)
    plt.plot(envdata["Temp"], envdata["Salinity"], ".")
    print(sps.linregress(envdata["Temp"], envdata["Salinity"]))
    plt.xticks([])
    plt.subplot(2, 2, 3)
    plt.plot(envdata["Temp"], envdata["Depth"], ".")
    print(sps.linregress(envdata["Temp"], envdata["Depth"]))
    plt.subplot(2, 2, 4)
    plt.plot(envdata["Salinity"], envdata["Depth"], ".")
    print(sps.linregress(envdata["Salinity"], envdata["Depth"]))
    plt.yticks([])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

# Find all the versions of the data
csvnames = sorted(glob("AllData-*.csv"), key=getfileno)
if len(csvnames) == 0:
    csvinname = "Matching Files"
    raise FileNotFoundError
# Pick out the one with the highest version number
csvinname = csvnames[-1]

# Get the useful rows from the data
alldata = pd.read_csv(csvinname)
print(alldata.shape[0], "total frames")
alldata = alldata.loc[::750]
print(alldata.shape[0], "surveyed frames")
alldata = alldata.loc[(alldata["Unusable"] == 0) & (alldata["ScaleOK"] > 0)]
print(alldata.shape[0], "usable surveyed frames")

sftbtmmask = (alldata["SoftBtm"] == 1)
hrdbtmmask = (alldata["HardBtm"] == 1)
clammask = (alldata["Clams"] == 1)
bactmask = (alldata["Bacteria"] == 1)
btmmasks = [sftbtmmask, hrdbtmmask, clammask, bactmask]
shrimpmask = (alldata["Shreemp"] > 0)

if "habitatconditions" in plotids:
    plt.boxplot([alldata[btmmask]["Depth"] for btmmask in btmmasks])
    plt.show()
    plt.boxplot([alldata[btmmask]["Temp"] for btmmask in btmmasks])
    plt.show()
    plt.boxplot([alldata[btmmask]["Salinity"] for btmmask in btmmasks])
    plt.show()

# (a/b + c/d + ...)/n is not necessarily the same as (a + c + ...)/(b + d + ...)
averageshrimpdensity = sum(alldata["Shreemp"]) / sum(alldata["AreaEstimate"])
exclsoftbtmshrimpdensity = (sum(alldata.loc[sftbtmmask & ~hrdbtmmask]["Shreemp"])
                            / sum(alldata.loc[sftbtmmask & ~hrdbtmmask]["AreaEstimate"]))
softbtmshrimpdensity = sum(alldata.loc[sftbtmmask]["Shreemp"]) / sum(alldata.loc[sftbtmmask]["AreaEstimate"])
bothshrimpdensity = (sum(alldata.loc[sftbtmmask & hrdbtmmask]["Shreemp"])
                     / sum(alldata.loc[sftbtmmask & hrdbtmmask]["AreaEstimate"]))
hardbtmshrimpdensity = sum(alldata.loc[hrdbtmmask]["Shreemp"]) / sum(alldata.loc[hrdbtmmask]["AreaEstimate"])
exclhardbtmshrimpdensity = (sum(alldata.loc[hrdbtmmask & ~sftbtmmask]["Shreemp"])
                            / sum(alldata.loc[hrdbtmmask & ~sftbtmmask]["AreaEstimate"]))
print("all: {}".format(averageshrimpdensity),
      "soft: {} | {} | {} | {} | {} :hard".format(exclsoftbtmshrimpdensity,
                                                  softbtmshrimpdensity,
                                                  bothshrimpdensity,
                                                  hardbtmshrimpdensity,
                                                  exclhardbtmshrimpdensity),
      "Shrimp per m^2", sep="\n")

print(list(alldata.columns))
print(alldata.loc[sftbtmmask].shape[0],
      alldata.loc[hrdbtmmask].shape[0],
      alldata.loc[sftbtmmask & hrdbtmmask].shape[0])

if "shrimpbottoms" in plotids:
    plt.subplot(1, 2, 1)
    plt.boxplot([np.log1p(alldata.loc[sftbtmmask]["Shreemp"] / alldata.loc[sftbtmmask]["AreaEstimate"]),
                 np.log1p(alldata.loc[hrdbtmmask]["Shreemp"] / alldata.loc[hrdbtmmask]["AreaEstimate"])])
    plt.subplot(1, 2, 2)
    plt.hist([np.log1p(alldata.loc[sftbtmmask]["Shreemp"] / alldata.loc[sftbtmmask]["AreaEstimate"]),
              np.log1p(alldata.loc[hrdbtmmask]["Shreemp"] / alldata.loc[hrdbtmmask]["AreaEstimate"])])
    plt.show()
    print(sps.ttest_ind(np.log1p(alldata.loc[sftbtmmask]["Shreemp"] / alldata.loc[sftbtmmask]["AreaEstimate"]),
                        np.log1p(alldata.loc[hrdbtmmask]["Shreemp"] / alldata.loc[hrdbtmmask]["AreaEstimate"])))
