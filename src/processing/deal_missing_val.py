INPUT_DIR='../../data/processed/'
OUTPUT_DIR='../../data/processed/'


import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder


# # Load data


data = pd.read_csv(INPUT_DIR+'data_remain_missing_val.csv')
data.head()


pd.Series(data.isnull().sum())[pd.Series(data.isnull().sum()) > 0].sort_values(ascending=False)


# # Number of USB 2.0 Ports


data['Number of USB 2.0 Ports'].value_counts()


# Có 2 giá trị vô lý là 10 và 25 (laptop ko thể có nhiều cổng USB như v) --> loại bỏ nó (vì ko rõ thực tế như thế nào)


impos_ind = data[(data['Number of USB 2.0 Ports']==25) | (data['Number of USB 2.0 Ports']==10)].index
data = data.drop(impos_ind)
data['Number of USB 2.0 Ports'].value_counts()


fillna_arr = np.random.choice(np.arange(len(data['Number of USB 2.0 Ports'].value_counts().index)), 
                              size=len(data[data['Number of USB 2.0 Ports'].isna()]['Number of USB 2.0 Ports']), 
                              p=data['Number of USB 2.0 Ports'].value_counts().values/sum(data['Number of USB 2.0 Ports'].value_counts()))


def fill_specific_na(x):
    global i
    if not (pd.isnull(x)):
        return x
    i+=1
    return fillna_arr[i]


i=-1
data['Number of USB 2.0 Ports'] = data['Number of USB 2.0 Ports'].apply(fill_specific_na)
data['Number of USB 2.0 Ports'].value_counts()


# Điền giá trị khiếm khuyết dựa trên tỷ lệ các giá trị.


# # Number of USB 3.0 Ports


data['Number of USB 3.0 Ports'].value_counts().sort_index()


fillna_arr = np.random.choice(np.arange(len(data['Number of USB 3.0 Ports'].value_counts().sort_index())), 
                              size=len(data[data['Number of USB 3.0 Ports'].isna()]['Number of USB 3.0 Ports']), 
                              p=data['Number of USB 3.0 Ports'].value_counts().sort_index().values/sum(data['Number of USB 3.0 Ports'].value_counts().values))


i=-1
data['Number of USB 3.0 Ports'] = data['Number of USB 3.0 Ports'].apply(fill_specific_na)
data['Number of USB 3.0 Ports'].value_counts()


# # old_price and price


data['price']


data.dropna(subset=['price'], inplace=True)


# Vì price là giá trị dự đoán --> ko thể tự ý điền giá trị thay thế --> xóa dòng.


nan_ind = data[data['old_price'].isna()].index
data.loc[nan_ind, 'old_price'] = data.loc[nan_ind, 'price']


data['old_price'].isnull().sum()


# Vì 'old_price' là giá tiền thời điểm trước đó của laptop (giá laptop có thay đổi về sau) nên giá trị NaN tức là giá tiền laptop không thay đổi, chỉ cần gán lại đúng giá trị biến 'price' tương ứng.


# # Optical Drive type


print(data['Optical Drive Type'].isna().sum())
data['Optical Drive Type'].value_counts()


data['Optical Drive Type'] = data['Optical Drive Type'].replace(regex=['no.*', '^(?!no).*'], value=['No', 'Yes'])
data['Optical Drive Type'].value_counts()


fillna_arr = np.random.choice([0, 1], 
                              size=len(data[data['Optical Drive Type'].isna()]['Optical Drive Type']), 
                              p=data['Optical Drive Type'].value_counts().values/sum(data['Optical Drive Type'].value_counts().values))


i=-1
data['Optical Drive Type'] = data['Optical Drive Type'].apply(fill_specific_na).replace(to_replace=[0,1], value=['No', 'Yes'])
data['Optical Drive Type'].value_counts()


# # Hard Drive type


drop_val = ['MultiMediaCard', 'UFS', 'SSHD', 'Hard', 'Flash', 'flash_memory_solid_state']
drop_index = data[(data['Hard Drive Type'].apply(str).apply(str.isdigit)) | (data['Hard Drive Type'].isin(drop_val))].index


data = data.drop(drop_index).replace('Solid', 'SSD').replace('hybrid', 'Hybrid').replace(regex='[E-e].[M-m][C-c]', value='eMMC')
data['Hard Drive Type'].value_counts()


fillna_arr = np.random.choice(np.arange(len(data['Hard Drive Type'].value_counts().index)), 
                              size=len(data[data['Hard Drive Type'].isna()]['Hard Drive Type']), 
                              p=data['Hard Drive Type'].value_counts().values/sum(data['Hard Drive Type'].value_counts().values))

i=-1
data['Hard Drive Type'] = data['Hard Drive Type'].apply(fill_specific_na).replace(to_replace=np.arange(len(data['Hard Drive Type'].value_counts().index)), 
                                                                                        value=data['Hard Drive Type'].value_counts().index)
data['Hard Drive Type'].value_counts()


# Vì chênh lệch số lượng quá lớn giữa các giá trị --> điền giá trị theo tỷ lệ xuất hiện của các giá trị.


# # Hard Drive size


data['Hard Drive Size'].isnull().sum()


# Vì dung lượng phần cứng là những con số cố định nên khi điền vào giá trị khiếm khuyết:
# + Không áp dụng phương pháp nội suy.
# + Không dùng giá trị mean.


data['Hard Drive Size'] = data['Hard Drive Size'].fillna(data.groupby('Hard Drive Type')['Hard Drive Size'].transform('median')).ffill()
data['Hard Drive Size']


# Vì dung lượng phần cứng là số cố định nên ta dùng phương thức transform của groupby pandas. Thông qua gom nhóm ta lấy giá trị trung vị điền vào giá trị khiếm khuyết.


# # Memory type


data['Memory Type'].value_counts()


drop_ind = data[data['Memory Type'].isin(['A8', 'DIMM', 'GDDR6'])].index
data.drop(drop_ind, inplace=True)


fillna_arr = np.random.choice(np.arange(len(data['Memory Type'].value_counts().index)), 
                              size=len(data[data['Memory Type'].isna()]['Memory Type']), 
                              p=data['Memory Type'].value_counts().values/np.sum(data['Memory Type'].value_counts().values))

i=-1
data['Memory Type'] = data['Memory Type'].apply(fill_specific_na).replace(to_replace=np.arange(len(data['Memory Type'].value_counts().index)), 
                                                                            value=data['Memory Type'].value_counts().index)
data['Memory Type'].value_counts()


# # Memory Size


data['Memory Size'].value_counts()


data['Memory Size'] = data['Memory Size'].fillna(data.groupby('Memory Type')['Memory Size'].transform('median'))
data['Memory Size']


# # Memory Speed


data['Memory Speed'] = data['Memory Speed'].fillna(data.groupby(['Memory Size', 'Memory Type'])['Memory Speed'].transform('mean')).ffill()
data['Memory Speed'].isnull().sum()


# Dựa theo dung lượng và loại bộ nhớ mà điền giá trị khiếm khuyết bằng trị giá trung vị.


# # Customer reviews


data['Customer Reviews'] = data['Customer Reviews'].interpolate(method='linear').round(1)
data['Customer Reviews']


# Sử dụng phương pháp nội suy để suy đoán các giá trị khiếm khuyết.


# # Graphics Coprocessor


drop_ind = data[data['Graphics Coprocessor']=='1'].index
data.drop(drop_ind, inplace=True)


data['Graphics Coprocessor']


encode = LabelEncoder()
temp = data[['Graphics Coprocessor']].astype("str").apply(encode.fit_transform)
final = temp.where(~data.isna(), data)
final


imputer = KNNImputer(n_neighbors=10)
final['new'] = imputer.fit_transform(final)
final['final'] = encode.inverse_transform(final['new'].map(int))
final


data['Graphics Coprocessor'] = final['final']
data['Graphics Coprocessor'].isnull().sum()


# # Graphics Coprocessor perf


data['Graphics Coprocessor perf'].isna().sum()


data['Graphics Coprocessor perf'] = data.groupby('Graphics Coprocessor')['Graphics Coprocessor perf'].apply(lambda x: x.interpolate()).bfill()
data['Graphics Coprocessor perf'].isnull().sum()


# # Processor


data['Processor'].value_counts()


encode = LabelEncoder()
temp = data[['Processor']].astype("str").apply(encode.fit_transform)
final = temp.where(~data.isna(), data)
final


imputer = KNNImputer(n_neighbors=10)
final['new'] = imputer.fit_transform(final)
final['final'] = encode.inverse_transform(final['new'].map(int))
final


data['Processor'] = final['final']
data['Processor'].isnull().sum()


# # Processor rank


data['Processor rank'].value_counts()


data['Processor rank'] = data.groupby('Processor')['Processor rank'].apply(lambda x: x.interpolate()).bfill()
data['Processor rank'].isnull().sum()


# # Best Seller rank


data['Best Sellers Rank'].isna().sum()


data.dropna(subset=['Best Sellers Rank'], inplace=True)


# # Laptop type


data['Laptop type'].value_counts()


fillna_arr = np.random.choice([0, 1], 
                              size=len(data[data['Laptop type'].isna()]['Laptop type']), 
                              p=data['Laptop type'].value_counts().values/sum(data['Laptop type'].value_counts().values))


i=-1
data['Laptop type'] = data['Laptop type'].apply(fill_specific_na).replace(to_replace=[0,1], value=['Traditional', '2 in 1'])
data['Laptop type'].value_counts()


# # Operating System


data['Operating System'].value_counts()


data['Operating System'] = data['Operating System'].fillna('Uninstalled')
data['Operating System'].value_counts()


# # Screen size


pd.Series(data['Screen Size'].value_counts())[pd.Series(data['Screen Size'].value_counts()) > 0]


nan_ind = data[data['Screen Size'] < 10].index
data.drop(nan_ind, inplace=True)


data['Screen Size'] = data.groupby('Brand')['Screen Size'].transform('median').ffill()
data['Screen Size']


# # Date First Available


len(data[data['Date First Available'].isna()].index)


data.dropna(subset=['Date First Available'], inplace=True)


# # Final check NaN


data.isnull().sum()


data.to_csv(OUTPUT_DIR+'final_data.csv')


