import pandas as pd
from discretization import *

file_name = 'UCI_CAD.csv'

if __name__ == '__main__':
    data = pd.read_csv(file_name, encoding='utf-8')
    features = list(data.columns[:-1])

    print(' featurtes of UCI_CAD', features)

    dataSR = squareRootChoice(data, features)
    dataSR.to_csv('UCI_CAD_SR.csv', index=False)

    dataSF = SturgesFormula(data, features)
    dataSF.to_csv('UCI_CAD_SF.csv', index=False)

    dataRR = RiceRule(data, features)
    dataRR.to_csv('UCI_CAD_RR.csv', index=False)

    dataDF = DoaneFormula(data, features)
    dataDF.to_csv('UCI_CAD_DF.csv', index=False)

    dataSN = ScottNormalReferenceRule(data, features)
    dataSN.to_csv('UCI_CAD_SN.csv', index=False)
