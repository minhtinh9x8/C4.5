import pandas as pd
import numpy as np
import math as mth

# Process Function

def countValues(data):
    if len(data)==0: return 0
    classes=data.unique()
    res={}
    for i in classes:
        count=0
        for j in data:
            if j==i:
                count+=1 
        res.update({i:count})
    return res

def countAttribute(X_col, y_train):
    X=pd.DataFrame()
    X['attribute']=X_col
    X['decision']=y_train
    res={}
    for i in X['attribute'].unique():
        val={}
        for j in X['decision'].unique():
            count=countValues(X['decision'][X['attribute']==i])
            val.update({j:count.get(j)})
        res.update({i:val})
    return res

def tranformNum(data):
    ts=data
    res=[]
    col=ts.columns
    temp=pd.DataFrame()
    for i in range(len(col[:-1])):
        temp[col[i]]=ts[col[:(i+1)]].sum(axis=1)
        temp['<%d' %col[i]]=ts[col[(i+1):]].sum(axis=1)
    return temp

def tranform(X_col, y_train):
    df=pd.DataFrame(countAttribute(X_col, y_train)).fillna(0)
    return df[df.columns.sort_values()]

# 2 Entropy, Gain, SplitInfo, Gain Ratio

def entropy(data):
    ent=0
    try:
        s=data.sum().sum()
        if s == 0: return ent
        for i in data.sum(axis=1).values:
            if i != 0: log=mth.log((i)/s, 2)
            else:log=0
            ent-=(i/s)*log
    except:
        s=data.sum()
        if s == 0: return ent
        for i in data.values:
            if i != 0: log=mth.log((i)/s, 2)
            else:log=0
            ent-=(i/s)*log
    return ent

def gain(X_tranform, entropyDecision):
    s=0
    for i in X_tranform.columns:
        if X_tranform.sum().sum()==0: s=0
        s+=float(entropy(X_tranform[i])*(X_tranform[i].sum()/X_tranform.sum().sum()))
    return entropyDecision-s
    

def splitInfo(data):   
    lenght=data.sum().sum()
    res=0
    if lenght==0: res=0
    for i in data.columns:
        val=data[i].sum().sum()
        res-=(val/lenght)*mth.log(val/lenght,2)
    return res

def gainRatio(gain, splitInfo):
    if splitInfo==0:
        return 0
    return gain/splitInfo

# 3 Gain Ratio for Num, get Max Gain Ratio

def getGainRatioNum(data, entropyDecision):
    res={}
    col=data.columns
    maxGain=0
    split=0
    for i in col[:-1]:
        name=[]
        name.append([i, '<%d' %i])
        X=tranformNum(data)[name[0]]
        gain_X=gain(X,entropyDecision)
        if gain_X>maxGain:
            col=i
            maxGain=gain_X
            split=splitInfo(X)
    return maxGain, split, col

def getResuft(X_train, y_train):
    col = X_train.columns
    entropyD=entropy(tranform(y_train, y_train))
    table={}
    
    for i in col:
        X=tranform(X_train[i], y_train)
        if X_train[i].dtype not in[str, object]:
            gainX, splitX, cX = getGainRatioNum(X, entropyD)
            gainRatioX = gainRatio(gainX, splitX)            
            table.update({i:{'col': i, 'val': cX, 'gain': gainX, 'gain ratio': gainRatioX}})
        else:
            gainX = gain(X, entropyD)
            splitX = splitInfo(X)
            gainRatioX = gainRatio(gainX, splitX)
            table.update({i:{'col': i, 'val': X.columns.tolist(), 'gain': gainX, 'gain ratio': gainRatioX}})
            
    maxG=-99999
    for i in table:
        gainR = table.get(i).get('gain ratio')
        col = table.get(i).get('col')
        val = table.get(i).get('val')
        tdt = table.get(i).get('type data')
        if gainR > maxG :
            maxG = gainR
            if str(val).isnumeric(): 
                c = '%s <=%d' %(col, int(val))
                print('yes')
            else: 
                c = col
                print('no')
            v = val
    return c, v

# 4 Fit and Show Node

def fit(X_train, y_train, decision='', note=''):
    col, val = getResuft(X_train, y_train)
    oldCol=col.split(' ')[0]
    X_train[y_train.name] = y_train
    
    if oldCol != col:
        X_train[oldCol]=X_train[oldCol].apply(lambda x: x <= val)
        X_train.rename(columns={oldCol: col}, inplace=True)
    for i in X_train[col].unique():
        new=X_train[X_train[col]==i]
        if new[y_train.name].nunique()>1:
            new_train=new.drop(columns=[y_train.name, col])
            new_test=new[y_train.name]
#             decision+='%s : %s -> %s' %(col, i, fit(new_train, new_test))
            note+='\n%s:%s, %s' %(col, i, fit(new_train, new_test))
        else:
#             decision+='%s : %s => %s; ' %(col, i, new[y_train.name].unique()[0])
            note+='\n%s:%s, %s' %(col, i, new[y_train.name].unique()[0])
                       
    return note

def showNode(n):
    dt=[]
    for i in n.split('\n'):
        dt.append(i.split(','))
    df=pd.DataFrame(dt, columns=['Columns', 'D']).drop(0)
    df['Condition']=df['Columns'].apply(lambda x: x.split(':')[0])
    df['Condition']=df['Condition'].apply(lambda x: x.split(' ')[-1])
    df['Values']=df['Columns'].apply(lambda x: x.split(':')[-1])
    df['Columns']=df['Columns'].apply(lambda x: x.split(':')[0])
    df['Columns']=df['Columns'].apply(lambda x: x.split(' ')[0])
    df['Decision']=df['D']
    df.drop(columns='D', inplace=True)
    return df

# II. Test
def main():
    data = pd.read_csv('./continuous_data.csv')
    X_train = data.drop(columns=['ID', 'Risk'])
    y_train=data['Risk']

    getResuft(X_train, y_train)

    data.info()

    model=fit(X_train, y_train)
    print(showNode(model))

if __name__ == "__main__":
    main()