import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

df = pd.read_csv('Iris.csv')
df.drop('Id',axis=1,inplace=True)

# Countplot
sns.countplot(x='Species',data=df)
plt.show()

# Barplot
sns.barplot(x='SepalLengthCm', y='Species', data=df, hue='Species', palette='inferno', legend=False)
plt.show()

sns.barplot(x='PetalLengthCm', y='Species', data=df, hue='Species', palette='inferno', legend=False)
plt.show()

# Scatterplot
sns.scatterplot(x = 'SepalLengthCm', y = 'SepalWidthCm' , data = df , hue = 'Species', palette = 'inferno' , s = 60)
plt.show()

sns.scatterplot(x = 'PetalLengthCm', y = 'PetalWidthCm' , data = df , hue = 'Species', palette = 'inferno' , s = 60)
plt.show()

# Pairplot
sns.pairplot(df, hue='Species', palette='Dark2')
plt.show()

# Boxplot
df.boxplot(by="Species", figsize=(12, 6))
plt.show()
