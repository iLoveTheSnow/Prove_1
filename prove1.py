from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pand

# S1
iris = datasets.load_iris()

# Show the data (the attributes of each instance)
print(iris.data)

# Show the target values (in numeric format) of each instance
print(iris.target)

# Show the actual target names that correspond to each number
print(iris.target_names)

# S2
colHead = "Sepal_Length Sepal_Width Petal_Length Pedal_Width".split()
datFrame = pand.DataFrame(iris.data, columns=colHead)
target = iris.target

data_train, data_test, target_train, target_test = train_test_split(datFrame, target, test_size=0.2)

# print(data_train.shape, target_train.shape)
# print(data_test.shape, target_test.shape)

# S3
classifier = GaussianNB()
model = classifier.fit(data_train, target_train)

# S4
target_predicted = model.predict(data_test)

print(target_predicted)
print("Percentage of Sklearn Corrent:")
print(100 * model.score(data_test, target_test))

# S5
class HardCodedClassifier:

    def __init__(self):
        pass

    def fit(self, data, target):
        return HardCodedClassifier()

    def prediction(self, test_data):
        target = []
        # x = 0
        for x in test_data:
            target.append(0)
        return target

    def percentScore(self, data_test, target_test):
        total = 0
        correct = 0
        # x = 0
        for x in data_test:
            total += 1

        i = 0
        while i < total:
            if data_test[i] == target_test[i]:
                correct += 1
            i += 1

        return float(correct / total)

BlankClassifier = HardCodedClassifier()

BlankModel = BlankClassifier.fit(data_train, target_train)
BlankPredicted = BlankModel.prediction(data_test)
print("Percentage of HardCoded Correct:")
print(100 * BlankModel.percentScore(BlankPredicted, target_test))
