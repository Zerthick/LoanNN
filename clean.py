import pandas as pd

#train_data = np.genfromtxt('credit_train.csv', delimiter=',')
df = pd.read_csv('credit_train.csv', usecols=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
l = pd.read_csv('credit_train.csv', usecols=[2])
print(df)
features = train_data[2:]