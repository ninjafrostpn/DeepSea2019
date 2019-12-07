import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

timetosec = np.vectorize(lambda x: (((x.hour * 60) + x.minute) * 60) + x.second)

# Read data in from the xlsx file as a pandas DataFrame (remove metadata in first 6 rows)
df = pd.read_excel("SOES6008_coursework_DATA.xlsx", skiprows=6)
df.columns = ["ActualTime", "VideoTime", "Lat", "Lon", "Dist", "Depth", "Temp", "Salinity"]
df = df.assign(VideoTimeSecs=timetosec(df["VideoTime"]))

# Plot vs Time
fig, ax1 = plt.subplots()
depthline = plt.plot(df["VideoTimeSecs"], df["Depth"], "b-", lw=0.5)[0]
ax1.set_ylabel("Depth (m)")
ax1.set_xlabel("Video Time (s)")
ax1.set_ylim(1520, 1360)
ax1.set_xlim(0, max(df["VideoTimeSecs"]))
ax2 = ax1.twinx()
templine = ax2.plot(df["VideoTimeSecs"], df["Temp"], "r-", lw=0.5)[0]
ax2.set_ylabel("Temperature (°C)", rotation=-90, labelpad=12)
ax2.set_ylim(0, 1)
ax3 = ax1.twinx()
salline = ax3.plot(df["VideoTimeSecs"], df["Salinity"], "g-", lw=0.5)[0]
ax3.set_ylabel("Salinity", rotation=-90, labelpad=12)
ax3.set_ylim(34.2, 34.8)
ax3.spines["right"].set_position(("axes", 1.2))
ax3.legend([depthline, templine, salline], ["Depth (m)", "Temperature (°C)", "Salinity"], loc="lower left")
fig.tight_layout()
plt.savefig("DepthTempSalinity over Time", dpi=200)
plt.show()

# Plot vs Dist
fig, ax1 = plt.subplots()
depthline = plt.plot(df["Dist"], df["Depth"], "b-", lw=0.5)[0]
ax1.set_ylabel("Depth (m)")
ax1.set_xlabel("Track Distance (m)")
ax1.set_ylim(1520, 1360)
ax1.set_xlim(0, max(df["Dist"]))
ax2 = ax1.twinx()
templine = ax2.plot(df["Dist"], df["Temp"], "r-", lw=0.5)[0]
ax2.set_ylabel("Temperature (°C)", rotation=-90, labelpad=12)
ax2.set_ylim(0, 1)
ax3 = ax1.twinx()
salline = ax3.plot(df["Dist"], df["Salinity"], "g-", lw=0.5)[0]
ax3.set_ylabel("Salinity", rotation=-90, labelpad=12)
ax3.set_ylim(34.2, 34.8)
ax3.spines["right"].set_position(("axes", 1.2))
ax3.legend([depthline, templine, salline], ["Depth (m)", "Temperature (°C)", "Salinity"], loc="lower left")
fig.tight_layout()
plt.savefig("DepthTempSalinity over Dist", dpi=200)
plt.show()
