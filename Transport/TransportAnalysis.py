# ------------------------------
# Tashkent Bus Network Analysis
# ------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# 1. Load the data
# ------------------------------
import pandas as pd

df = pd.read_csv(
    "Transport.csv",
    sep=None,           # auto-detect separator
    engine='python',    # flexible parser
    encoding='latin1',  # fallback for Windows CSVs
    on_bad_lines='skip' # skip malformed lines
)

print(df.head())
print(df.info())

print("Data loaded successfully!")
print(df.head())
print(df.info())

# ------------------------------
# 2. Clean & preprocess
# ------------------------------
# Convert numeric columns
df['Routing distance'] = pd.to_numeric(df['Routing distance'], errors='coerce')
df['Intermediate interval'] = pd.to_numeric(df['Intermediate interval'], errors='coerce')

# Function to convert working hours like "06:00-22:00" to total hours
def get_working_hours(duration_str):
    try:
        start, end = duration_str.split('-')
        h1, m1 = map(int, start.split(':'))
        h2, m2 = map(int, end.split(':'))
        return (h2 + m2/60) - (h1 + m1/60)
    except:
        return None

df['Working_hours_weekday'] = df['Working hours (working day)'].apply(get_working_hours)
df['Working_hours_saturday'] = df['Working hours (Saturday)'].apply(get_working_hours)
df['Working_hours_sunday'] = df['Working hours (Sunday)'].apply(get_working_hours)

# ------------------------------
# 3. Basic statistical insights
# ------------------------------
print("\n--- Basic Insights ---")
print("Average route distance:", round(df['Routing distance'].mean(), 2), "km")
print("Average interval between buses:", round(df['Intermediate interval'].mean(), 2), "minutes")
print("Most common bus model:", df['Bus Model'].mode()[0])

# Routes per filial
routes_per_filial = df.groupby('Filial')['Route number'].nunique().sort_values(ascending=False)
print("\nNumber of routes per filial:")
print(routes_per_filial)

# ------------------------------
# 4. Visualizations
# ------------------------------

sns.set(style="whitegrid")

# 4.1 Distribution of Route Distances
plt.figure(figsize=(10,6))
sns.histplot(df['Routing distance'], bins=20, kde=True, color='skyblue')
plt.title("Distribution of Bus Route Distances")
plt.xlabel("Distance (km)")
plt.ylabel("Number of Routes")
plt.show()

# 4.2 Average working hours per filial (weekdays)
avg_hours = df.groupby('Filial')['Working_hours_weekday'].mean().sort_values()
plt.figure(figsize=(10,6))
avg_hours.plot(kind='bar', color='lightgreen')
plt.title("Average Working Hours per Filial (Weekdays)")
plt.ylabel("Hours")
plt.xlabel("Filial")
plt.show()

# 4.3 Bus model popularity
bus_counts = df['Bus Model'].value_counts()
plt.figure(figsize=(10,6))
sns.barplot(x=bus_counts.values, y=bus_counts.index, palette="viridis")
plt.title("Bus Model Distribution")
plt.xlabel("Number of Routes")
plt.ylabel("Bus Model")
plt.show()

# 4.4 Boxplot: Intermediate interval by filial
plt.figure(figsize=(12,6))
sns.boxplot(x='Filial', y='Intermediate interval', data=df, palette="pastel")
plt.title("Intermediate Interval Distribution by Filial")
plt.ylabel("Interval (minutes)")
plt.xlabel("Filial")
plt.show()

# 4.5 Compare working hours: Weekday vs Saturday vs Sunday
plt.figure(figsize=(12,6))
df_melt = df.melt(id_vars=['Route number'],
                  value_vars=['Working_hours_weekday','Working_hours_saturday','Working_hours_sunday'],
                  var_name='Day_Type', value_name='Hours')
sns.boxplot(x='Day_Type', y='Hours', data=df_melt, palette="Set2")
plt.title("Working Hours Comparison: Weekday vs Saturday vs Sunday")
plt.ylabel("Hours")
plt.xlabel("Day Type")
plt.show()

# ------------------------------
# 5. Optional: Export summary statistics
# ------------------------------
summary = df.groupby('Filial').agg({
    'Route number': 'nunique',
    'Routing distance': ['mean','min','max'],
    'Intermediate interval': ['mean','min','max'],
    'Working_hours_weekday': 'mean',
    'Working_hours_saturday': 'mean',
    'Working_hours_sunday': 'mean'
}).round(2)

summary.columns = ['Routes','Distance_mean','Distance_min','Distance_max',
                   'Interval_mean','Interval_min','Interval_max',
                   'Workday_hours_avg','Saturday_hours_avg','Sunday_hours_avg']

summary.to_csv("bus_network_summary.csv")
print("\nSummary statistics saved to bus_network_summary.csv")
print(summary)
