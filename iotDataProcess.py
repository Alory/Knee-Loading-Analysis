"""
@Description : This script is for process the raw data from the iot sensor.
@Author : Wangchao
@Date : 2018.3.22
"""
import os
import json
from functools import reduce

import pandas as pd
import numpy as np

imuCols = 'LLACx\tLLACy\tLLACz\tLLGYx\tLLGYy\tLLGYz\tLMACx\tLMACy\tLMACz\tLMGYx\tLMGYy\tLMGYz\tRLACx\tRLACy\tRLACz\tRLGYx\tRLGYy\tRLGYz\tRMACx\tRMACy\tRMACz\tRMGYx\tRMGYy\tRMGYz'

addr2pos = {"90BDB8B2-4737-1EB3-8AC7-756943596524": "LL", "7DFEC5A2-697F-482F-C6A8-9A0450ECC674": "LM", \
            "9DBD3CB0-12E9-D9F8-823A-EEAEA7A840D1": "RL", "8D6C8805-B13E-9F56-5B88-1DC63407869F": "RM", \
            "ADF9BB24-6649-8CC1-AF67-8AACB4F146EC": "RM", "92F37785-2E73-2E79-0F6F-856BDEC44D29":"RL"}
currentDir = os.getcwd()


# The function to process the iot data
def iotDataProcess(path, fileName):
    if (fileName[-3:] != 'txt' or fileName[0:8] == 'modified'):  # filename[0:6] == 'static' or
        pass
    else:
        rawData = open(path + fileName, 'r');  # raw data file
        rawDataFile = rawData.readlines()
        processedData = open(path + 'modified-' + fileName, 'w')  # processed data file

        processedData.write('address\tsensor\tx\ty\tz\t\ttime\n');
        try:
            for line in rawDataFile:
                tempData = json.loads(line)
                index = 1000 if (tempData['sensor'] == 'accelerometer') else 1;
                x = float(tempData['value']['x']) * index
                y = float(tempData['value']['y']) * index
                z = float(tempData['value']['z']) * index
                pos = addr2pos[tempData['address']]
                strValue = str(x) + '\t' + str(y) + '\t' + str(z)
                processedData.write(
                    pos + '\t' + tempData['sensor'] + '\t' + strValue + '\t' + str(
                        tempData['time']) + '\n')
        finally:
            rawData.close()
            processedData.close()


def iotDataProcess2(path, filename):
    dataCols = dict(LLACx=[], LLACy=[], LLACz=[], LLGYx=[], LLGYy=[], LLGYz=[], LMACx=[], LMACy=[], LMACz=[],
                    LMGYx=[], LMGYy=[], LMGYz=[], RLACx=[], RLACy=[], RLACz=[], RLGYx=[], RLGYy=[], RLGYz=[], RMACx=[],
                    RMACy=[], RMACz=[], RMGYx=[], RMGYy=[], RMGYz=[])

    if (filename[-3:] != 'txt' or filename[0:8] == 'modified'):  # filename[0:6] == 'static' or
        pass
    else:
        rawData = open(path + filename, 'r')  # raw data file
        rawDataFile = rawData.readlines()

        try:
            for line in rawDataFile:
                tempData = json.loads(line)
                index = 1000 if (tempData['sensor'] == 'accelerometer') else 1
                x = float(tempData['value']['x']) * index
                y = float(tempData['value']['y']) * index
                z = float(tempData['value']['z']) * index
                pos = addr2pos[tempData['address']]

                sensor = 'AC' if (index == 1000) else 'GY'
                dataCols[pos + sensor + 'x'].append(x)
                dataCols[pos + sensor + 'y'].append(y)
                dataCols[pos + sensor + 'z'].append(z)

        finally:
            rawData.close()

            values = dataCols.values()
            colNum = min(list(map(lambda x: len(x), values)))
            dataList = list(map(lambda x: dataCols[x][0:colNum], dataCols))
            data = np.array(dataList)
            data = data.transpose()
            np.savetxt(path + 'modified-' + filename,data,header=imuCols,delimiter='\t',fmt ='%.2f',comments='')

            #data = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dataCols.items()]))
            # data = pd.DataFrame.from_dict(dataCols, orient='index').transpose()
            #data.to_csv(path + 'modified-' + filename, index=False, sep='\t')


def recursiveFile(filePath):
    fileList = os.listdir(filePath)
    for file in fileList:
        if (os.path.isdir(file)):
            recursiveFile(file)
        else:
            iotDataProcess2(filePath + '\\', file)


if __name__ == '__main__':
    recursiveFile(currentDir)
