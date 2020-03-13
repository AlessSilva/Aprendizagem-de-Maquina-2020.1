#!/usr/bin/env python
# coding: utf-8

# # Atividade 2 - Regressão Linear
# ### Nome: Alessandro Souza Silva, Matrícula: 399941

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


# ### Implementação da Regressão Linear e das Métricas de Avaliação (RSS,RSE,R2,MAE,MSE)

# In[10]:


class RegressaoLinearSimples(object):
    
    def __init__(self):
        pass
    
    def fit(self, X, y):
        
        self.X = X
        self.y = y
        
        x_mean = np.mean(self.X)
        y_mean = np.mean(self.y)
        
        self.b1 = np.sum((self.X - x_mean)*(self.y - y_mean))
        self.b1 = self.b1/np.sum((self.X - x_mean)**2)
        
        self.b0 = y_mean - (self.b1*x_mean)
        
    def predict(self, _X):
        
        self._y = self.b0 + self.b1*_X
        return self._y

class RegressaoLinearMultipla(object):
    
    def __init__(self):
        pass
    
    def fit(self, X, y):
        
        bias = np.ones((X.shape[0],1))
        self.X = np.concatenate( (bias,X),axis=1)
        self.y = y
        
        self.b_ = self.X.T.dot( self.X )
        self.b_ = np.linalg.inv(self.b_)
        self.b_ = self.b_.dot(self.X.T)
        self.b_ = self.b_.dot(self.y)
        
    def predict(self, _X):
        
        bias = np.ones((_X.shape[0],1))
        _X = np.concatenate( (bias,_X),axis=1)
        
        _y = np.dot(_X, self.b_.T)
        
        return _y

def RSS( y_true, y_predict ):
    return np.sum( np.power((y_true-y_predict),2) )

def RSE( y_true, y_predict ):  
    n = y_true.shape[0]
    rss = RSS(y_true, y_predict)
    return np.power(( rss/(n-2) ),0.5)

def R2( y_true, y_predict ):
    y_mean = np.mean(y_true)
    tss = np.sum( np.power((y_true-y_mean),2) )
    rss = RSS( y_true, y_predict )
    return 1 - ( rss/tss )

def MAE( y_true, y_predict ):
    n = y_true.shape[0]
    return np.sum( abs(y_true-y_predict) ) / n

def MSE( y_true, y_predict ):
    n = y_true.shape[0]
    return np.sum( np.power(y_true-y_predict,2) ) / n


# ### Leitura dos dados utilizando a biblioteca pandas

# In[3]:


import pandas as pd
data = pd.read_fwf("Data/housing.data")


# In[4]:


data.head()


# ### Embaralhando as amostras com seus valores alvo

# In[5]:


data = data.sample(frac=1, random_state=40)
data.head()


# ### Utilizando apenas a variável LSTAT como atributo preditor e a variável MEDV como atributo alvo

# In[6]:


X = data.LSTAT
y = data.MEDV


# ### Dividindo o conjunto de dados em 80% para treino e 20% para teste utilizando o sklearn

# In[7]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4)


# ### Criando e treinando um modelo de Regressão Linear Simples

# In[8]:


rls = RegressaoLinearSimples()

rls.fit(X_train,y_train)


# ### Reportando MSE e R2 score para o conjunto de treino

# In[11]:


y_pred_train = rls.predict(X_train)
print("Conjuto de treino")
print("MSE: ",MSE(y_train,y_pred_train))
print("R2: ",R2(y_train,y_pred_train))


# ### Reportando MSE e R 2 score para o conjunto de teste

# In[12]:


y_pred_test = rls.predict(X_test)
print("Conjuto de teste")
print("MSE: ",MSE(y_test,y_pred_test))
print("R2: ",R2(y_test,y_pred_test))


# ### Plotando um gráfico com LSTAT no eixo X e MEDV no eixo Y onde é apresentado tanto os dados originais (conjunto inteiro) como pontos quanto a reta da regressão linear

# In[13]:


plt.scatter(X,y,c='y')
plt.plot(X_test,y_pred_test,c='g')
plt.grid()
plt.show()


# ### Plotando um gráfico onde os valores preditos para o conjunto de treino estão no eixo X e os valores alvo originais estão no eixo Y. Também é plotado a reta Y=X para comparação.

# In[14]:


plt.scatter(y_pred_train,y_train,c='b')
plt.plot(range(35),range(35),c='r')
plt.show()


# ### Adicionando o termo LSTAT2 e LSTAT3 ao conjunto de dados e refazendo toda  a análise anterior

# In[15]:


data["LSTAT2"] = [ np.power(x,2) for x in data.LSTAT]
data["LSTAT3"] = [ np.power(x,3) for x in data.LSTAT]
data.head()


# #### Análise com o termo LSTAT2

# In[16]:


X = data.loc[:,["LSTAT","LSTAT2"]]
y = data["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4)

rlm = RegressaoLinearMultipla()

rlm.fit(X_train,y_train)


# In[17]:


y_pred_train = rlm.predict(X_train)
print("Conjuto de treino")
print("MSE: ",MSE(y_train,y_pred_train))
print("R2: ",R2(y_train,y_pred_train))
print()

y_pred_test = rlm.predict(X_test)
print("Conjuto de teste")
print("MSE: ",MSE(y_test,y_pred_test))
print("R2: ",R2(y_test,y_pred_test))


# In[18]:


X_test = X_test.sort_values(by=['LSTAT'])
y_pred_test = rlm.predict(X_test)

plt.scatter(X.LSTAT,y,c='y')
plt.plot(X_test.LSTAT,y_pred_test,c='g')
plt.grid()
plt.show()


# In[19]:


plt.scatter(y_pred_train,y_train,c='b')
plt.plot(range(45),range(45),c='r')
plt.show()


# #### Análise com o termo LSTA3

# In[20]:


X = data.loc[:,["LSTAT","LSTAT2","LSTAT3"]]
y = data["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4)

rlm = RegressaoLinearMultipla()

rlm.fit(X_train,y_train)


# In[21]:


y_pred_train = rlm.predict(X_train)
print("Conjuto de treino")
print("MSE: ",MSE(y_train,y_pred_train))
print("R2: ",R2(y_train,y_pred_train))
print()

y_pred_test = rlm.predict(X_test)
print("Conjuto de teste")
print("MSE: ",MSE(y_test,y_pred_test))
print("R2: ",R2(y_test,y_pred_test))


# In[22]:


X_test = X_test.sort_values(by=['LSTAT'])
y_pred_test = rlm.predict(X_test)

plt.scatter(X.LSTAT,y,c='y')
plt.plot(X_test.LSTAT,y_pred_test,c='g')
plt.grid()
plt.show()


# In[23]:


plt.scatter(y_pred_train,y_train,c='b')
plt.plot(range(45),range(45),c='r')
plt.show()

