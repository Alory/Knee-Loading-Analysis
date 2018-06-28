from dataProcessing import *
out = 'kam2cali/'
iotout = 'kam2allinfo/'

tempIotcols = ['LLACx', 'LLACy', 'LLACz', 'LLGYx', 'LLGYy', 'LLGYz'
        , 'LMACx', 'LMACy', 'LMACz', 'LMGYx', 'LMGYy', 'LMGYz'
        , 'RLACx', 'RLACy', 'RLACz', 'RLGYx', 'RLGYy', 'RLGYz'
        , 'RMACx', 'RMACy', 'RMACz', 'RMGYx', 'RMGYy', 'RMGYz','mass']
allcols = ['LLACx','LLACy','LLACz','LLGYx','LLGYy','LLGYz','LMACx','LMACy','LMACz','LMGYx','LMGYy','LMGYz',
           'RLACx','RLACy','RLACz','RLGYx','RLGYy','RLGYz','RMACx','RMACy','RMACz','RMGYx','RMGYy','RMGYz',
           'mass','height','Lleglen','LkneeWid','LankleWid','Rleglen','RkneeWid','y']


iotSubjects = ['iot-S10','iot-S11','iot-S16','iot-S18','iot-S21','iot-S22','iot-S23','iot-S33','iot-S35','iot-S7','iot-S8','iot-S12']

if __name__ == '__main__':
    subname = 'iot-all'
    testList = tempIotcols[0:24]
    sensor = 'iot'# sensor in ['iot','norx']
    if(sensor == 'iot'):
        data = pd.read_csv(iotout + subname + '.txt', sep="\t")
        varData = data[testList]
        transData = pd.DataFrame([],columns=testList)
        for pos in testList:
            imuPos = iot2imuPos[iotCols.index(pos)]
            flag = iot2imuCols[imuPos]
            transData[imuPos] = flag*varData[pos]
    else:
        data = pd.read_csv(out + subname + '.txt', sep="\t")
        varData = data[testList]
        transData = varData
    corrCoef = np.corrcoef(transData,rowvar=0)

    pl.figure(figsize=(10, 6))
    ax = sns.heatmap(corrCoef, fmt=".2f", linewidths=.5, cmap='YlGnBu',annot=True)
    ax.yaxis.set_ticklabels(testList, rotation=0, ha='right', fontsize=9)
    ax.xaxis.set_ticklabels(testList, rotation=45, ha='right', fontsize=9)
    subtitle = 'IoT data' if (sensor == 'iot') else 'Noraxon data'
    pl.title(subtitle + '\nVariables correlation coefficient')
    pl.show()
