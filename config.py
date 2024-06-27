# path setup

DATA_PATH = '/media/imlab/HDD/gear_anomaly/4. Preprocessed Data/'
# DATA_PATH = '/media/imlab/HDD/'


# Columns of Interest
COI = ['Time', 'CylPres', 'CylPresAct', 
       'LatAccel', 'LongAccel', 'YawRate',
       'WhlSpdFL', 'WhlSpdFR', 'WhlSpdRL', 'WhlSpdRR', 
       'EngStat', 'BrkDep',
       'AccDep', 'EngRPM', 'TarGear', 'SAS', 
       'InhibitD', 'InhibitN',
       'InhibitP', 'InhibitR', 
       'VehSpdClu', 
       'BatSOC', 
       'HeadLampHigh',
       'HeadLampLow', 
       'IndLeft', 'IndRight', 
       'DriveMode', 
       'OutTemp',
       'FuelEconomy', 'HevMode', 'BrkAct', 'EngColTemp', 
       'ODO', 
    #    'Latitude', 'Longitude', 'GPSMode', 'SatelliteNum', 'Altitude', 
    #    'Yaw', 'Pitch', 'Roll', 'TrueNorth', 'NorthDeclination', 
    #    'Grid X', 'Grid Y', 'Temp',
    #    'Precipitation', 'EWwind', 'SNwind', 'Humidity', 'PrecipitationType',
    #    'WindDirection', 'WindSpeed', 'LinkID', 'Congestion', 'LinkDirection',
    #    'LinkRoadType', 'LinkLength', 'LinkPassTime', 'LinkSpeed',
       ]

# Columns of Inputs
Input_columns = ['Timestamp',
    'TarGear', 'LatAccel', 'LongAccel', 'YawRate', 'SAS', 'EngStat',
    'BrkDep', 'AccDep', 'EngRPM', 'WhlSpdFL', 'WhlSpdFR', 'WhlSpdRL',
    'WhlSpdRR', 'EngColTemp', 'VehSpdClu'
]