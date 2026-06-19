# Rishab-Goswami
Ml prediction salary 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from word2number import w2n
import re
from thefuzz import process
from sklearn.linear_model import LinearRegression
df = pd.read_excel('messy_employee_salary_data.xlsx')
# jin coloumn ka use nahi usko delete karna
drop_col = ['Emp_ID','Gender','Marital_Status','Remarks']
df.drop(drop_col, axis=1, inplace=True)
# experience ko clean karna
# experience or age ko clean karna
for col in ['Years_of_Experience','Age']:
  df[col] = df[col].replace(['Nan','None','Null','N/A','??',''], np.nan)
  df[col] = df[col].astype(str)
  df[col] = df[col].str.replace(r'[^0-9.\s]','',regex=True)
  df[col] = df[col].str.strip()
  df[col] = pd.to_numeric(df[col], errors='coerce')
# city ko clean karna
df['City'] = (df['City'].str.replace(r'[^a-z A-Z\s]','',regex=True).astype(str).str.strip().str.title())
standard_city = ['Jaipur','Gurugram','Bengaluru','Pune','Mumbai','Hyderabad','Chennai','New Delhi','Kolkata','Noida','Madras','Bombay','Calcutta']
invalid_values = {'unknown','ufknown','urknown','urknown','uaknown','nan','none','na','n/a','null','nana'}
def clean_city (city):
  if pd.isna(city):
    return 'unknown'
  if city.lower() in invalid_values:
    return 'unknown'
  result = process.extractOne(city, standard_city)
  if result:
    match = result[0]
    score = result[1]
    if score >= 90:
      return match
  return 'unknown'
df['City'] = df['City'].apply(clean_city)
# phone ko clean karna 
def clean_phone(phone):
  if pd.isna(phone):
    return np.nan
  degits = re.sub(r'\D','',phone)
  if degits.startswith('91') and len(degits) == 12:
    degits = degits[2:]
  elif degits.startswith('0') and len(degits) == 11:
    degits = degits[1:]
  if len(degits) == 10 and degits[0] in '6789':
    return degits
  return np.nan
df['Phone_Number'] = df['Phone_Number'].apply(clean_phone).fillna('Unknown')
# Name, Department, Education, Job_Title
for coloumn in ['Name', 'Department', 'Education', 'Job_Title']:
  junk = ['None','Null','N/A','Nan','??','']
  df[coloumn] = df[coloumn].astype(str).str.strip().str.title()
  df[coloumn] = df[coloumn].str.replace(r'[^a-z A-Z.\s]','', regex=True)
  df[coloumn] = df[coloumn].replace(junk, 'Unknown')
replace_map = {
    'Senior Analyst': 'Senior Analyst',
    'Sr Analyst': 'Senior Analyst',
    'Sr. Analyst': 'Senior Analyst',

    'Software Eng.': 'Software Engineer',
    'Software Engineer': 'Software Engineer',

    'Team Lead': 'Team Lead',
    'Team Leader': 'Team Lead',
    'Tl': 'Team Lead',

    'Trainee': 'Trainee',
    'Unknown': 'Unknown'
}

df['Job_Title'] = df['Job_Title'].replace(replace_map)
# Joining_Date ko clean karna
df['Joining_Date'] = pd.to_datetime(df['Joining_Date'], errors='coerce')
df['Joining_Date'] = df['Joining_Date'].bfill()
# salary ko clean karna 
multipliers = {
    'LPA' : 100000,
    'thousand' : 1000,
    'thousands' : 1000,
    'lakh' : 100000,
    'lakhs' : 100000,
    'crore' : 10_000_000,
    'crores' : 10_000_000
}
def clean_salary (x):
  try:
    return str(w2n.word_to_num(x))
  except:
    return x
df['Salary'] = df['Salary'].replace(['Nan','None','Null','N/A','??',''], np.nan)
working = df['Salary'].astype(str).str.strip().str.title()
working = working.apply(clean_salary)
working = working.str.replace('₹','',regex=False)
working = working.str.replace('rs.','',regex=False)
working = working.str.replace('INR','',regex=False)
working = working.str.replace(',','',regex=False)
nums = working.str.extract(r'(\d+(?:.\d+)?)')[0]
nums = pd.to_numeric(nums, errors='coerce')
mults = pd.Series(1.0, index=df.index)
for word, factor in multipliers.items():
  match_mask = working.str.contains(rf'{word}', case=False, na=False)
  mults[match_mask] = factor
df['Salary'] = nums * mults
# outlier ko clean kaerna
# duplicates remove 
df.drop_duplicates(inplace = True)
for cols in ['Age','Years_of_Experience','Performance_Rating','Salary']:
  df[cols] = pd.to_numeric(df[cols], errors='coerce')
  Q1 = df[cols].quantile(0.25)
  Q3 = df[cols].quantile(0.75)
  IQR = Q3 - Q1
  lower = Q1 - 1.5 * IQR
  upper = Q3 + 1.5 * IQR
  df[cols] = df[cols].clip(lower = lower, upper=upper)
for mis in ['Age','Years_of_Experience','Performance_Rating']:
  df[mis] = df[mis].fillna(df[mis].mean())
df['Salary'] = df['Salary'].fillna(df['Salary'].median())
plt.figure(figsize=(20,10))
sns.histplot(x='Salary', data=df, kde=True, color='orange')
plt.title('Salary Distribution')
plt.figure(figsize=(15,9))
sns.barplot(x='Department',y='Salary',data=df, palette='viridis')
plt.title('Department wise sales')
plt.show()
df = pd.get_dummies(df, columns = ['Department','Job_Title'], dtype = int)
df.head()
X = df[['Years_of_Experience','Department_It','Department_Marketing','Department_Mktg','Department_Operations', 'Department_Ops', 'Department_Sale', 'Department_Sales','Department_Tech', 'Department_Unknown','Job_Title_Engineer',	'Job_Title_Intern',	'Job_Title_Manager',	'Job_Title_Mgr', 'Job_Title_Sde',	'Job_Title_Senior Analyst',	'Job_Title_Software Engineer', 'Job_Title_Team Lead',	'Job_Title_Trainee',	'Job_Title_Unknown']]
y = df['Salary']
model = LinearRegression()
model.fit(X,y)
yof = int(input('Enter a Years_of_Experience :'))
job = input('Enter a job title (Engineer,	Intern,	Manager,	Mgr, Sde,	Senior Analyst,	Software Engineer, Team Lead,	Trainee,	Unknown) : ').lower()
title_engineer = 0
title_intern = 0
title_manager = 0
title_mgr = 0
title_sde = 0
title_senior_analyst = 0
title_software_engineer = 0
title_team_lead	= 0
title_trainee = 0
title_unknown = 0
if job == 'engineer':
  title_engineer = 1
elif job == 'intern':
  title_intern = 1
elif job == 'mangaer':
  title_mangaer = 1
elif job == 'mgr':
  title_mgr = 1
elif job == 'sde':
  title_sde = 1
elif job == 'senior_anaylst':
  title_senior_analyst = 1
elif job == 'software_engineer':
  title_software_engineer = 1
elif job == 'team_lead':
  title_team_lead = 1
elif job == 'trainee':
  title_trainee = 1
elif job == 'unknown':
  title_unknown = 1
department = input('Enter a deptarment (Department_It,Department_Marketing,Department_Mktg,Department_Operations,Department_Ops,Department_Sale,Department_Sales,Department_Sales,Department_Tech,Department_Unknown) : ').lower()
dept_it = 0
dept_marketing = 0
dept_mktg = 0
dept_operations = 0
dept_ops = 0
dept_sale = 0
dept_sales = 0
dept_tech = 0
dept_unknown = 0
if department == 'it':
  dept_it = 1
elif department == 'marketing':
  dept_marketing = 1
elif department == 'mktg':
  dept_mktg = 1
elif department == 'operations':
  dept_operations = 1
elif department == 'ops':
  dept_ops = 1
elif department == 'sale':
  dept_sale = 1
elif department == 'sales':
  dept_sales = 1
elif department == 'tech':
  dept_tech = 1
elif department == 'unknown':
  dept_unknown = 1
predict_salary = model.predict([[yof,dept_it,dept_marketing,dept_mktg,dept_operations,dept_ops,dept_sale,dept_sales,dept_tech,dept_unknown,title_engineer,title_intern,title_manager,title_mgr,title_sde,title_senior_analyst,title_software_engineer,title_team_lead,title_trainee,title_unknown]])
print(f'The predict {predict_salary[0]} is salary ')