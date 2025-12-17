import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# =========================
# 1. LOAD DATA
# =========================
# If your file is CSV → use read_csv
# df = pd.read_csv("vacancies.csv")

# For your uploaded file (JSON):
df = pd.read_json("vacancies.json")

print("Initial shape:", df.shape)

# =========================
# 2. BASIC CLEANING
# =========================
df.columns = df.columns.str.lower()

# Drop duplicates (same title + company + location)
df = df.drop_duplicates(subset=["title", "company", "location"])

# =========================
# 3. SALARY CLEANING
# =========================
df["salary_from"] = pd.to_numeric(df["salary_from"], errors="coerce")
df["salary_to"] = pd.to_numeric(df["salary_to"], errors="coerce")

# Average salary if both exist
df["salary_avg"] = df[["salary_from", "salary_to"]].mean(axis=1)

# =========================
# 4. CURRENCY NORMALIZATION
# =========================
# Approximate exchange rates (can be updated)
EXCHANGE = {
    "UZS": 1,
    "USD": 12500,
    "EUR": 13500
}

df["currency"] = df["currency"].fillna("UZS")

df["salary_uzs"] = df.apply(
    lambda x: x["salary_avg"] * EXCHANGE.get(x["currency"], 1)
    if pd.notna(x["salary_avg"]) else np.nan,
    axis=1
)

# =========================
# 5. ROLE NORMALIZATION
# =========================
def normalize_role(title):
    title = title.lower()

    if re.search(r"junior|intern|trainee|стажер", title):
        return "Junior"
    elif re.search(r"senior|lead|старш", title):
        return "Senior"
    elif re.search(r"middle", title):
        return "Middle"
    else:
        return "Not specified"

df["level"] = df["title"].apply(normalize_role)

def role_type(title):
    title = title.lower()

    if "bi" in title or "business intelligence" in title:
        return "BI / Analytics"
    if "data scientist" in title or "data science" in title:
        return "Data Science"
    if "analyst" in title:
        return "Data Analyst"
    if "engineer" in title:
        return "Data Engineer"
    return "Other"

df["role_type"] = df["title"].apply(role_type)

# =========================
# 6. LOCATION CLEANING
# =========================
df["location"] = df["location"].str.strip()
df["location"] = df["location"].replace({"Акташ (Узбекистан)": "Акташ"})

# =========================
# 7. INSIGHT 1: JOBS BY CITY
# =========================
plt.figure()
df["location"].value_counts().head(10).plot(kind="bar")
plt.title("Top 10 Cities by Job Count")
plt.xlabel("City")
plt.ylabel("Number of vacancies")
plt.tight_layout()
plt.show()

# =========================
# 8. INSIGHT 2: SALARY DISTRIBUTION
# =========================
plt.figure()
sns.histplot(df["salary_uzs"], bins=20, kde=True)
plt.title("Salary Distribution (UZS)")
plt.xlabel("Salary (UZS)")
plt.tight_layout()
plt.show()

# =========================
# 9. INSIGHT 3: AVERAGE SALARY BY LEVEL
# =========================
salary_by_level = (
    df.groupby("level")["salary_uzs"]
    .mean()
    .dropna()
    .sort_values(ascending=False)
)

print("\nAverage Salary by Level (UZS):")
print(salary_by_level)

salary_by_level.plot(kind="bar", title="Average Salary by Level (UZS)")
plt.ylabel("UZS")
plt.tight_layout()
plt.show()

# =========================
# 10. INSIGHT 4: ROLE DEMAND
# =========================
plt.figure()
df["role_type"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Role Demand Distribution")
plt.ylabel("")
plt.tight_layout()
plt.show()

# =========================
# 11. INSIGHT 5: COMPANIES THAT SHOW SALARY
# =========================
salary_visibility = (
    df.assign(has_salary=df["salary_uzs"].notna())
      .groupby("company")["has_salary"]
      .mean()
      .sort_values(ascending=False)
      .head(10)
)

print("\nTop companies by salary transparency:")
print(salary_visibility)

# =========================
# 12. SAVE CLEAN DATA
# =========================
df.to_csv("vacancies_cleaned.csv", index=False)
print("\nCleaned dataset saved as vacancies_cleaned.csv")
