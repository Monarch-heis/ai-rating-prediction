import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("test1.csv")
df.drop("SessionID",axis=1,inplace=True)
df.drop("SessionDate",axis=1,inplace=True)
df.drop("UsedAgain",axis=1,inplace=True)
lb = LabelEncoder()
df.head()
df["StudentLevel"] = lb.fit_transform(df["StudentLevel"])
df["Discipline"] = lb.fit_transform(df["Discipline"])

df["FinalOutcome"] = lb.fit_transform(df["FinalOutcome"])

df["TaskType"] = lb.fit_transform(df["TaskType"])

df.head()

x=df.drop('SatisfactionRating',axis='columns')
y=df['SatisfactionRating']

model=LinearRegression()

model.fit(x,y)

model.predict([[2,6,30,10,5,3,0]])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
print(x_train)

m = model.coef_

b = model.intercept_
print(b)

y_pred = model.predict(x_test)

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

from sklearn.metrics import mean_squared_error,mean_absolute_error
print(mean_squared_error(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))

st.title('A.I asisstant rating Predictor')
st.write("""
**Aliases**

**Student Level**
- High school = 0  
- Graduate = 1  
- Undergraduate = 2  

**Discipline**
- Biology = 0  
- Business = 1  
- Computer Science = 2  
- Engineering = 3  
- History = 4  
- Math = 5  
- Psychology = 6  

**Final Outcome**
- Assignment completed = 1  
- Confused = 2  
- Idea drafted = 3  

**Task Type**
- Brainstorming = 0  
- Coding = 1  
- Homework help = 2  
- Research = 3  
- Studying = 4  
- Writing = 5  
""")
studentlevel = st.number_input('Enter your Student level', min_value=0, max_value=2, value=1)
discipline = st.number_input('Enter the no. of feilds used', min_value=0.0, max_value=10.0, value=6.0)
sessionlengthmin = st.number_input('Enter Session lenght', min_value=0.0, max_value=100.0, value=10.0)
totalprompts = st.number_input('Total prompts', min_value=0, max_value=20, value=5)
tasktype = st.number_input('Nature of the task', min_value=0, max_value=5, value=0)
ai_assistancelevel = st.number_input('1–5 scale on how helpful the AI was perceived to be', min_value=0, max_value=5, value=1)
finaloutcome = st.number_input('no. of needed outcomes procured ', min_value=0, max_value=10, value=5)

df.head(5)

if st.button('Predict Rating'):
    pred = model.predict([[studentlevel,discipline,sessionlengthmin,totalprompts,tasktype,ai_assistancelevel,finaloutcome]])
    st.write('Prediction result (rating):', pred[0])

st.subheader("Training Dataset Preview")
st.write(df.head())
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
ax.set_title("Correlation Heatmap of All Features")
st.pyplot(fig)