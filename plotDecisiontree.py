from sklearn.externals import joblib
from sklearn import tree
import os
feature = ['LLACx','LLACy','LLACz','LLGYx','LLGYy','LLGYz','LMACx','LMACy','LMACz','LMGYx','LMGYy','LMGYz',
           'RLACx','RLACy','RLACz','RLGYx','RLGYy','RLGYz','RMACx','RMACy','RMACz','RMGYx','RMGYy','RMGYz',
           'age', 'mass', 'height', 'Lleglen', 'LkneeWid', 'Rleglen', 'LankleWid', 'RkneeWid', 'RankleWid',
            'gender_F', 'gender_M']

model = joblib.load('model/' + 'RandomForest-trees-5-depth-10iot-allData-L.model')

decisionTree = (model.estimators_)[0]
dotfile = open("dtree2.dot", 'w')
tree.export_graphviz(decisionTree, out_file = dotfile, feature_names = feature)
dotfile.close()
os.system("dot -Tpng D:.dot -o dtree2.png")