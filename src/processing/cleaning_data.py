#This file is exported from jupyter notebook
INPUT_DIR='data/raw/'
OUTPUT_DIR='data/processed/'


import pandas as pd
import regex as re
import numpy as np
import json


# # Input dữ liệu


df1 = pd.read_csv(INPUT_DIR+'laptop_list.csv')
df2 = pd.read_json(INPUT_DIR+'laptop_detail.json')
df=pd.merge(df1, df2, how='inner', on='link')  
# df1.shape, df2.shape, df.shape


# # Tiền xử lý


df=df.apply(lambda x: x.str.replace('\u200e', ''), axis=1)


for col in df.columns:
    try:
        df[col]=df[col].str.lower()
    except:
        pass


# ## cpu


df['Processor']=df['Processor'].str.replace(r'_', ' ')


dfsub=df[['name', 'description', 'Processor']]
cpu_extracted_by_regex=[]
cpu_regex=r'((mediatek|ryzen|celeron|xeon|intel|pentiumn|core|amd).{0,30}?(\w+\d{4}\w*|\w*\d{4}\w+)|snapdragon 7c( gen 2){0,1})'
k=0
for i in range(dfsub.shape[0]):
    found=re.search(
        cpu_regex, 
        dfsub.iloc[i]['name'],
        flags=re.IGNORECASE
    )
    if found:
        cpu_extracted_by_regex.append(found.group())
        #print(dfsub.iloc[i, 0].strip())
        #print(found.group())
    else:
        try:
            found2=re.search(
                cpu_regex, 
                dfsub.iloc[i]['description'],
                flags=re.IGNORECASE
            )
        except TypeError:
            pass
            found2=False
        finally:
            if found2:
                #print(dfsub.iloc[i, 0].strip())
                #print(found2.group())
                k+=1
                cpu_extracted_by_regex.append(found2.group())
            else:
                cpu_extracted_by_regex.append(np.nan)
                #print(df.iloc[i]['official link'])
                #print(dfsub.iloc[i]['name'])
                #print(dfsub.iloc[i]['Processor'])


df['Processor']=cpu_extracted_by_regex


df['Processor']=df['Processor'].str.replace('_', ' ')
#df['Processor']=df['Processor'].str.replace(r'core|pentinum|celeron', 'intel', regex=True)
#df['Processor']=df['Processor'].str.replace(r'ryzen', 'amd', regex=True)


cpu_df=pd.read_csv(INPUT_DIR+'cpu_rank.csv')
cpu_df_list=list(zip(cpu_df['CPU name'], cpu_df['CPU rank']))
cpu_df_list=[list(x) for x in cpu_df_list]
for i in range(len(cpu_df_list)):
    cpu_df_list[i].append(cpu_df_list[i][0].split()[-1])
cpu_df_list


#truong hop loi
#Intel Xeon E3-1246 v3
#amd ryzen 5 5500u vs amd ryzen 5 pro 5500u
def transform_cpu_name_and_rank(row):
    processor = row['Processor']
    if pd.notna(processor):
        """
        brand=re.findall(r'intel|amd|snapdragon|mediatek', processor)[0]
        model=re.findall('\w+$', processor)[0]
        found=False
        for x in cpu_df_list:
            if brand==x[2] and model==x[3]:
                row['Processor']=x[0]
                row['Processor rank']=x[1]
                found=True
                break
            if(found==False):
                row['Processor']=np.nan
                row['Processor rank']=np.nan
        return row
        """
        model = re.findall("\w+$", processor)[0]
        match1 = []
        for i, x in enumerate(cpu_df_list):
            if x[2] == model:
                match1.append(i)
        if len(match1) >= 2:
            found = (-1, -1)
            processor_wbw = processor.split()[:-1]
            for x in match1:
                f = 0
                for word in processor_wbw:
                    if word in cpu_df_list[x][0]:
                        f += 1
                if f > found[0]:
                    found = (f, x)
            pos = found[1]
        elif len(match1)==1:
            pos = match1[0]
        else:
            row["Processor"] = np.nan
            row["Processor rank"] = np.nan
            #print(row["Processor"])
            return row
        row["Processor"] = cpu_df_list[pos][0]
        row["Processor rank"] = cpu_df_list[pos][1]
    return row


df=df.apply(transform_cpu_name_and_rank, axis=1)


df['Processor'].apply(lambda x: re.findall(r'^(\w*)\s', x)[0] if pd.notna(x) else x).value_counts()


# ## card


with open(INPUT_DIR+'card-performance.json', 'r') as rf:
    card_perf_df=json.load(rf) 
card_list=[x[0] for x in card_perf_df]


from Levenshtein import ratio
"""
def LCS(s, t):
    n = len(s)
    m = len(t)
    s=' '+s
    t=' '+t

    f = [[0 for i in range(m + 1)] for j in range(n+1)]
     
    for i in range(1,n + 1):
        for j in range(1,m + 1):
            if(s[i - 1] == t[j - 1]):
                f[i][j] = f[i - 1][j - 1] + 1
            else:
                f[i][j] = max(f[i-1][j], f[i][j-1])
    return (f[n][m]+n+m)/(n+m)
"""
def find_relative_string(s, slist):
    found=[-1, -1] #max value,position
    for i in range(len(slist)):
        acc=ratio(s, slist[i])
        if found[0]<acc:
            found[0]=acc
            found[1]=i
    return found


def transform_graphic(row):
    card = row['Graphics Coprocessor']
    if pd.notna(card):
        card=card.lower()
        acc, pos=find_relative_string(card, card_list)
        if acc>0.5:
            #print(card)
            #print(card_perf_df[pos][0])
            row["Graphics Coprocessor"] = card_perf_df[pos][0]
            row["Graphics Coprocessor perf"] = card_perf_df[pos][1]
        else:
            row["Graphics Coprocessor"] = np.nan
            row["Graphics Coprocessor perf"] = np.nan
    return row


df=df.apply(transform_graphic, axis=1)


# ## weight


df['weight']=df['Item Weight'].str.split().str[0]
df['weight']= df['weight'].astype(float)
df['weight'] *= 0.45359237


# ## Hard Drive Type/size


df['Hard Drive Type']=df['Hard Drive'].str.split().str[2]
df['HD Size']= df['Hard Drive'].str.extract('(\d+)').fillna(np.nan).astype(float)
df['HD_Type']=df['Hard Drive'].str.split().str[1] 
df['HD_Type']=df['HD_Type'].apply({'gb':1,'tb':1024}.get)
df['Hard Drive Size'] = df['HD Size'] * df['HD_Type']


def nan_1_time_cate(serie):
    valuecounts=serie.value_counts()
    unkeep=list(valuecounts[valuecounts==1].index)
    return serie.replace(unkeep, np.nan)


df['Hard Drive Type']=nan_1_time_cate(df['Hard Drive Type'])
drop_index = df[(df['Hard Drive Type'].apply(str).apply(str.isdigit))].index
df.drop(drop_index, inplace=True)


df['Hard Drive Type']=df['Hard Drive Type'].replace('solid', 'ssd')


df['Hard Drive Type']=df['Hard Drive Type'].replace(regex=r'.*([^ssd][^hdd][^hybrid]).*', value='other')


df['Hard Drive Type'].value_counts()


# ## screen size


df['Screen Size']=df['Standing screen display size'].str.split().str[0]
df['Screen Size']=nan_1_time_cate(df['Screen Size'])


# ## memory size/type/speed


df['RAM']=df['RAM'].replace({'lpddr 5':'lpddr5'}, regex=True)
df['Memory Size'] =  df['RAM'].str.extract('(\d+)').fillna(np.nan)
df['Memory Type'] =df['RAM'].str.split().str[-1]
df['Memory Type']=df['Memory Type'].replace(['gb', 'tb'], np.nan)


nan_1_time_cate(df['Memory Type']).value_counts()


df['Memory Speed']=df['Memory Speed'].replace({' ghz':'',' mhz':'/1000'}, regex=True).dropna().apply(pd.eval)
df['Memory Speed']=df['Memory Speed'].apply(lambda x: x/1000 if x>1000 else x)


# ## customer review


df['Customer Reviews']=df['Customer Reviews'].str.split().str[0]


# ## price


df['old_price']=df['old_price'].str.extract('(\d+.\d+)')
df['old_price']=df['old_price'].str.replace(',', '')
df['price']=df['price'].str.extract('(\d+.\d+)')
df['price']=df['price'].str.replace(',', '')


# ## laptop type


def extract_type(row):
    if(pd.notna(row)):
        x = re.findall(r'traditional|2 in 1', row)
        if x:
            return x[0]

df['Laptop type']=df['Best Sellers Rank'].apply(extract_type)


df['Laptop type'].value_counts()


# ## best seller rank


df['Best Sellers Rank']=df['Best Sellers Rank'].str.extract(r'^#([\d,]+)')
df['Best Sellers Rank']=df['Best Sellers Rank'].str.replace(',', '')
df['Best Sellers Rank']=df['Best Sellers Rank'].astype(np.float64)


sellerrank=df['Best Sellers Rank'].dropna().tolist()
sellerrank.sort()
sellerrank_dict={}
dem=0
pre=-1
for x in sellerrank:
    if x > pre: 
        dem+=1
        sellerrank_dict[x]=dem
    if(pd.notna(x)):
        pre=x



df['Best Sellers Rank']=df['Best Sellers Rank'].replace(sellerrank_dict)


# ## Date First Available


df['Date First Available'] = pd.to_datetime(df['Date First Available'])


# ## operating system


df['Operating System']=df['Operating System'].str.replace(r'.*(windows|win).*', 'windows', regex=True)
df['Operating System']=df['Operating System'].str.replace(r'.*chrome.*', 'chrome os', regex=True)
df['Operating System']=df['Operating System'].str.replace(r'.*[^chrome os][^windows].*', '#', regex=True)
df['Operating System']=df['Operating System'].replace('#', np.nan)


# ## brand


possible_brand=list(df['Brand'].dropna().unique())
possible_brand=[x for x in possible_brand if x!='intel']


def transfom_brand(row):
    for brand in possible_brand:
        if brand in str(row['name']) or brand in str(row['Brand']):
            row['Brand']=brand
            break
    if(pd.isna(row['Brand'])):
        row['Brand']=re.findall(r'^[\d\s]*(\w+)', row['name'])[0]
    return row
df=df.apply(transfom_brand, axis=1)


df['Brand']=df['Brand'].replace('hewlett packard', 'hp')
df['Brand']=df['Brand'].replace('intel', np.nan)


df['Brand']=nan_1_time_cate(df['Brand'])


df['Brand'].value_counts()


# ## laptop purpose


def extract_purpose(row):
    x = re.findall(r'gaming', row)
    if x:
        return 'gaming'
    else:
        return 'general'

df['Laptop purpose']=df['name'].apply(extract_purpose)


# ## Number of USB 2.0 Ports


df['Number of USB 2.0 Ports']=nan_1_time_cate(df['Number of USB 2.0 Ports'])


# ## Optical Drive Type


df['Optical Drive Type'] = df['Optical Drive Type'].replace(regex=['no.*', '^(?!no).*'], value=['no', 'yes'])


# # last step


# check the number of acutal values in per column
cols=df.columns
remain_df=[]
for col in cols:
    remain=df.shape[0]-df[col].isna().sum()
    remain_df.append([col, remain, df[col].sample(3)])
remain_df.sort(reverse=True, key=lambda x: x[1])
remain_df


cols_to_keep=[
'name',
'Brand',
'Best Sellers Rank',
'weight',
'Laptop type',
'Laptop purpose',
'Screen Size',
'Hard Drive Size',
'Hard Drive Type',
'Memory Speed',
'Memory Size',
'Memory Type',
'Processor',
'Processor rank',
'Graphics Coprocessor',
'Graphics Coprocessor perf',
'Optical Drive Type',
'Operating System',
'Number of USB 3.0 Ports',
'Number of USB 2.0 Ports',
'Date First Available',
'Customer Reviews',
'old_price',
'price',
]
df=df[cols_to_keep]


for col in df.columns:
    try:
        df[col] = df[col].astype(np.float64)
    except Exception:
        continue


df['Optical Drive Type']


df=df.drop_duplicates(subset=[
'Brand',
'weight',
'Laptop type',
'Laptop purpose',
'Screen Size',
'Hard Drive Size',
'Hard Drive Type',
'Memory Speed',
'Memory Size',
'Memory Type',
'Processor',
'Processor rank',
'Graphics Coprocessor',
'Graphics Coprocessor perf',
'Optical Drive Type',
'Operating System',
'Number of USB 3.0 Ports',
'Number of USB 2.0 Ports',
])


df=df.sort_values(by=['price'])


df.to_csv(OUTPUT_DIR+'data_remain_missing_val.csv', index=False)


