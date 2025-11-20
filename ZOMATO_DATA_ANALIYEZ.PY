
#   @ = ZOMATO – Data Analysis Questions

# 1 What type of restaurant do the majority of customers order from?

# 2 How many votes has each type of restaurant received from customers?

# 3 What are the ratings that the majority of restaurants have received?

# 4 Zomato has observed that most couples order most of their
 # food online. What is their average spending on each order?

# 5 Which mode (online or offline) 
 # has received the maximum rating?

# 6 Which type of restaurant received more offline orders, 
 #so that Zomato can provide customers with some good offers?



import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt

df = pd.read_csv("Zomato data .csv")
print(df.head(10))

#convet the data type of column - ret
def handalRate (value):
    value = str(value).split('/')
    value = value[0]
    return float (value)
df['rate'] = df['rate'].apply(handalRate)
print(df.head())

#infomation all data
print(df.info())

#show the resturent 
sns.countplot(x=df['listed_in(type)'],color="r")
plt.xlabel("type of resturent")
plt.show()

# 1 ANS =  the majority of customers order of Dining resturent



grupe_data = df.groupby('listed_in(type)')['votes'].sum()

result = pd.DataFrame({'votes': grupe_data})

plt.figure(figsize=(12,6))
plt.plot(result.index, result['votes'], c='green', marker='o')

plt.xlabel("Type of restaurant", color="red", fontsize=20)
plt.ylabel("Votes", color="red", fontsize=20)
plt.show()
# ans 2 = conclusion - dinnig resturants has recived maximum votes


plt.hist(df['rate'],bins=5)
plt.title("rating distribution ")
plt.show()
 #ANS 3 = conclusion - the majority resturant receved rating frome 3.5 to 5



couple_data = df['approx_cost(for two people)']
sns.countplot(x=couple_data)
plt.show()

# ans 4 =  The majorit of coupels of preferr resturant with an approxi 
   # cost of 300 rupesh


plt.figure(figsize=(6,6))
sns.boxenplot(x='online_order' , y='rate',data= df)
plt.show()

# ANS 5 = offline order received lower rating in comparison to online oredr

pivot_table = df.pivot_table(
    index='listed_in(type)',
    columns='online_order',
    aggfunc='size'
)

sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='d')
plt.title("heatmap")
plt.xlabel("online order")
plt.ylabel("Listed in (type)")
plt.show()
