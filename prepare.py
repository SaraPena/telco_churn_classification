import pandas as pd
import numpy as np

import env
import acquire
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

query = '''SELECT * 
           FROM customers
           JOIN contract_types USING (contract_type_id)
           JOIN internet_service_types USING (internet_service_type_id)
           JOIN payment_types using (payment_type_id)'''

db = 'telco_churn'

df = acquire.get_data_from_mysql(query,db)

# set index to customer_id:

df.set_index('customer_id', inplace = True)
df.head()
df.columns

df.replace(to_replace = " ", value = np.nan, inplace = True)

# df.info()
# After we replace ' ' (space) with np.nan we see that there are 7032 non-null values for total_charges. All other variables have 7043

# df[df.total_charges.isna()][['total_charges','tenure', 'monthly_charges']]
# We see from the dataframe where total_charges.isna() has all tenure of 0. We can assume to mean that there is total_charges for these customers yet.
# Because these customers has no tenure, they have not had a opportunity to churn or not churn.
# Our data set is going to have enough over 7000 rows, so we will drop these rows. 

df.dropna(axis = 0, inplace = True)

# Look at the new dataset:
# df.info()

# Create a new column named 'tenure_years' this converts our time of months into a measurement of years.
df['tenure_years'] = df.tenure/12

# Create single variable representing the information from phone_service, and multiple lines

phone_service = [0 if i == 'No' else 1 for i in df.phone_service]
multiple_lines = [1 if i == 'Yes' else 0 for i in df.multiple_lines]

df['phone_and_multi_line'] = [phone_service[i] + multiple_lines[i] for i in range(len(phone_service))]

# Do the same thing using dependents and partners:
df[(df.partner == 'No') & (df.dependents == 'Yes')]

partner_dependents = []
for i in range(len(df.partner)):
    if df.partner[i] == 'No' and df.dependents[i] == 'No':
        partner_dependents.append(0)
    elif df.partner[i] == 'Yes' and df.dependents[i] == 'No':
        partner_dependents.append(1)
    elif df.partner[i] == 'No' and df.dependents[i] == 'Yes':
        partner_dependents.append(2)
    elif df.partner[i] == 'Yes' and df.dependents[i] == 'Yes':
        partner_dependents.append(3)

df['partner_dependents'] = partner_dependents

# Look at data types of each column:
# df.info()
# total_charges is an object datatype. We need to change this to float using astype()

df.total_charges = df.total_charges.astype('float')

#df.info()

# Create one_hot_encoder for payment, internet_service, and contract_type id's.
# payment ohe
ohe = OneHotEncoder(sparse = False, categories = 'auto')
payment_ohe = ohe.fit_transform(df[['payment_type_id']])

# payment_ohe.shape

labels_payment = list(np.array(df.payment_type.value_counts().index))

payment_df = pd.DataFrame(payment_ohe, columns = labels_payment, index = df.index)
payment_df.info()

# internet serivce ohe
ohe = OneHotEncoder(sparse = False, categories = 'auto')
internet_ohe = ohe.fit_transform(df[['internet_service_type_id']])

labels_internet = list(df.internet_service_type.value_counts().sort_index().index)

internet_df = pd.DataFrame(internet_ohe, columns = labels_internet, index = df.index)

# Contact type ohe

ohe = OneHotEncoder(sparse = False, categories = 'auto')
contract_ohe = ohe.fit_transform(df[['contract_type_id']])

labels_contract = list(df.contract_type.value_counts().sort_index().index)

contract_df = pd.DataFrame(contract_ohe, columns = labels_contract, index = df.index)

# Join one hot encoder dataframes to main dataframe
ohe_df = df.join([payment_df, internet_df, contract_df])



# Create dataframe to explore variables contract type, internet type, payment type, monthly_charges, tenure, total_charges
#df = df[['payment_type_id','internet_service_type_id','tenure', 'monthly_charges', 'total_charges', 'churn', 'contract_type', 'internet_service_type','payment_type']]

# prep_df = df


# Next split the data into train, and test
train, test = train_test_split(df, test_size = .3, random_state = 123, stratify = df.churn)

train = train
test = test


train_ohe, test_ohe = train_test_split(ohe_df, test_size = .3, random_state = 123, stratify = ohe_df.churn)
#train_ohe.head()

train_ohe = train_ohe
test_ohe = test_ohe
