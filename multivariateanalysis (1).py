import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("loan_prediction.csv")
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
sns.countplot(x='Gender', hue='Married', data=df)
plt.title('Gender vs Married')
plt.subplot(1, 3, 2)
sns.countplot(x='Education', hue='Self_Employed', data=df)
plt.title('Education vs Self Employed')
plt.subplot(1, 3, 3)
sns.countplot(x='Property_Area', hue='Loan_Amount_Term', data=df)
plt.title('Property Area vs Loan Amount Term')
plt.tight_layout()
plt.show()