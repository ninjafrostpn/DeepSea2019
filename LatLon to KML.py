import pandas as pd

# Template for the whole kml file
blankdocument = ('<?xml version="1.0" encoding="UTF-8"?>'
                 '<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2" xmlns:kml="http://www.opengis.net/kml/2.2" xmlns:atom="http://www.w3.org/2005/Atom">'
                 '<Document>\n'
                 '    <name>{5}.kml</name>\n'
                 '    <StyleMap id="m_ylw-pushpin">\n'
                 '        <Pair>\n'
                 '            <key>normal</key>\n'
                 '            <styleUrl>#s_ylw-pushpin</styleUrl>\n'
                 '        </Pair>\n'
                 '        <Pair>\n'
                 '            <key>highlight</key>\n'
                 '            <styleUrl>#s_ylw-pushpin_hl</styleUrl>\n'
                 '        </Pair>\n'
                 '    </StyleMap>\n'
                 '    <Style id="s_ylw-pushpin">\n'
                 '        <IconStyle>\n'
                 '            <scale>1.1</scale>\n'
                 '            <Icon>\n'
                 '                <href>http://maps.google.com/mapfiles/kml/pushpin/ylw-pushpin.png</href>\n'
                 '            </Icon>\n'
                 '            <hotSpot x="20" y="2" xunits="pixels" yunits="pixels"/>\n'
                 '        </IconStyle>\n'
                 '        <LineStyle>\n'
                 '            <color>{1}</color>\n'
                 '            <width>{2}</width>\n'
                 '        </LineStyle>\n'
                 '    </Style>\n'
                 '    <Style id="s_ylw-pushpin_hl">\n'
                 '        <IconStyle>\n'
                 '            <scale>1.3</scale>\n'
                 '            <Icon>\n'
                 '                <href>http://maps.google.com/mapfiles/kml/pushpin/ylw-pushpin.png</href>\n'
                 '            </Icon>\n'
                 '            <hotSpot x="20" y="2" xunits="pixels" yunits="pixels"/>\n'
                 '        </IconStyle>\n'
                 '        <LineStyle>\n'
                 '            <color>{1}</color>\n'
                 '            <width>{2}</width>\n'
                 '        </LineStyle>\n'
                 '    </Style>\n'
                 '    <Placemark>\n'
                 '        <name>{0}</name>\n'
                 '        <description>{3}</description>\n'
                 '        <styleUrl>#m_ylw-pushpin</styleUrl>\n'
                 '        <LineString>\n'
                 '            <tessellate>1</tessellate>\n'
                 '            <altitudeMode>absolute</altitudeMode>\n'
                 '            <coordinates>\n'
                 '                {4}\n'
                 '            </coordinates>\n'
                 '        </LineString>\n'
                 '    </Placemark>\n'
                 '{6}'
                 '</Document>\n'
                 '</kml>')

# Template for any pins added to the trace
blankpin = ('  <Placemark>\n'
            '    <name>{1}</name>'
            '    <description>{2}</description>\n'
            '    <styleUrl>#pointstyle</styleUrl>\n'
            '    <Point>\n'
            '      <altitudeMode>absolute</altitudeMode>'
            '      <coordinates>{0}</coordinates>\\nn'
            '    </Point>\n'
            '  </Placemark>\n')


def fromXLSX(filepath):
    # Read data in from the xlsx file as a pandas DataFrame (remove metadata in first 6 rows)
    df = pd.read_excel(filepath, skiprows=6)
    df.columns = ["ActualTime", "VideoTime", "Lat", "Lon", "Dist", "Depth", "Temp", "Salinity"]
    # Each point is formatted (Lon, Lat, Depth, Name, Description)
    # Name is see on the pin on the map, Description only when one clicks on the pin
    pts = []
    for i in range(len(df["Lat"])):
        print(df["VideoTime"][i], -df["Depth"][i])
        # Points actually placed at altitude above terrain, but with same depth profile
        pts.append([df["Lon"][i], df["Lat"][i], 10 + max(df["Depth"]) - df["Depth"][i],
                    df["VideoTime"][i],
                    "{}\n{}N,{}E\n{}m depth\n{}°C\n{} salinity".format(df["ActualTime"][i],
                                                                       df["Lat"][i], df["Lon"][i], df["Depth"][i],
                                                                       df["Temp"][i], df["Salinity"][i])])
    return pts


def toKML(pts, name="", filename=None, col=(255, 165, 0), width=1, desc=""):
    if filename is None:
        filename = name
    if filename.endswith(".txt"):
        filename = filename[:-4]
    filename.replace(".", "-")
    hexcol = "ff" + hex((((col[0] << 8) + col[1]) << 8) + col[2])[2:]
    # Data for the depth profile trace
    ptstring = " ".join([",".join([str(float(i)) for i in pt[:3]])
                         for pt in pts])
    # Data for the pins on the trace; every 50th record is given a pin with data on it
    pinstring = "".join([blankpin.format(",".join([str(i) for i in pt[:3]]), pt[3], pt[4])
                         for pt in pts[::50]])
    return blankdocument.format(name, hexcol, width, desc, ptstring, filename, pinstring)


def convertfile(filepath):
    # Convert the xslx data into the kml display
    KMLdata = toKML(fromXLSX(filepath))
    # Then save the thing
    with open(filepath[:-4] + "kml", 'w', newline='') as KMLfile:
        KMLfile.write(KMLdata)


convertfile("SOES6008_coursework_DATA.xlsx")
