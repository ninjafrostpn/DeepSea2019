import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

timetosec = np.vectorize(lambda x: (((x.hour * 60) + x.minute) * 60) + x.second)

# Read data in from the xlsx file as a pandas DataFrame (remove metadata in first 6 rows)
df = pd.read_excel("SOES6008_coursework_DATA.xlsx", skiprows=6)
df.columns = ["ActualTime", "VideoTime", "Lat", "Lon", "Dist", "Depth", "Temp", "Salinity"]

# Plot vs Time
fig, ax = plt.subplots()
templine = ax.plot(timetosec(df["VideoTime"]), df["Temp"], "r-", lw=0.5)[0]
ax.set_ylabel("Temperature (°C)")
ax.set_xlabel("Video Time (s)")
ax.set_ylim(0, 1)
ax = ax.twinx()
salline = plt.plot(timetosec(df["VideoTime"]), df["Salinity"], "b-", lw=0.5)[0]
ax.set_ylabel("Salinity", rotation=-90, labelpad=12)
ax.set_ylim(34, 34.8)
ax.legend([templine, salline], ["Temperature (°C)", "Salinity"], )
fig.tight_layout()
plt.savefig("TempSalinity over Time", dpi=200)
plt.show()

# Plot vs Dist
fig, ax = plt.subplots()
templine = ax.plot(df["Dist"], df["Temp"], "r-", lw=0.5)[0]
ax.set_ylabel("Temperature (°C)")
ax.set_xlabel("Track Distance (m)")
ax.set_ylim(0, 1)
ax = ax.twinx()
salline = plt.plot(df["Dist"], df["Salinity"], "b-", lw=0.5)[0]
ax.set_ylabel("Salinity", rotation=-90, labelpad=12)
ax.set_ylim(34, 34.8)
ax.legend([templine, salline], ["Temperature (°C)", "Salinity"], )
fig.tight_layout()
plt.savefig("TempSalinity over Dist", dpi=200)
plt.show()
