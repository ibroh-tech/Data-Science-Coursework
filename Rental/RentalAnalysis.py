import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'

# Load and clean data
print("Loading data...")
df = pd.read_csv('olx_rentals_tashkent_full.csv')

df_clean = df.copy()
df_clean['currency'] = df_clean['currency'].fillna('у.е.')
df_clean['price_numeric'] = pd.to_numeric(df_clean['price'], errors='coerce')

df_clean = df_clean[df_clean['price_numeric'].notna()]
df_clean = df_clean[df_clean['price_numeric'] > 0]
df_clean = df_clean[df_clean['price_numeric'] < 10000000]

df_clean['rooms'] = pd.to_numeric(df_clean['rooms'], errors='coerce')
df_clean['area_m2'] = pd.to_numeric(df_clean['area_m2'], errors='coerce')
df_clean['floor'] = pd.to_numeric(df_clean['floor'], errors='coerce')
df_clean['total_floors'] = pd.to_numeric(df_clean['total_floors'], errors='coerce')

df_ue = df_clean[df_clean['currency'] == 'у.е.'].copy()
df_usd = df_clean[df_clean['currency'] == '$'].copy()
df_sum = df_clean[df_clean['currency'] == 'сум'].copy()
df_ue_with_area = df_ue[df_ue['area_m2'].notna()].copy()
df_ue_with_area['price_per_m2'] = df_ue_with_area['price_numeric'] / df_ue_with_area['area_m2']

print("✓ Data loaded and cleaned")


# ============================================================================
# DASHBOARD 1 - PRICE ANALYSIS
# ============================================================================
def show_dashboard_1():
    print("\n[Opening Dashboard 1: Price Analysis]")
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('TASHKENT RENTAL MARKET - PRICE ANALYSIS DASHBOARD',
                 fontsize=18, fontweight='bold', y=0.98)

    # 1.1 Histogram
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(df_ue['price_numeric'], bins=40, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Price (у.е.)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Frequency', fontweight='bold', fontsize=11)
    ax1.set_title('📊 Price Distribution (в у.е.)', fontweight='bold', fontsize=12)
    ax1.axvline(df_ue['price_numeric'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {df_ue["price_numeric"].mean():.0f}')
    ax1.axvline(df_ue['price_numeric'].median(), color='green', linestyle='--', linewidth=2,
                label=f'Median: {df_ue["price_numeric"].median():.0f}')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 1.2 Box Plot
    ax2 = plt.subplot(2, 3, 2)
    price_data = [df_ue['price_numeric'].dropna(), df_usd['price_numeric'].dropna(), df_sum['price_numeric'].dropna()]
    bp = ax2.boxplot(price_data, labels=['у.е.', '$', 'сум'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral', 'lightgreen']):
        patch.set_facecolor(color)
    ax2.set_ylabel('Price', fontweight='bold', fontsize=11)
    ax2.set_title('📦 Price by Currency', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    # 1.3 Pie Chart
    ax3 = plt.subplot(2, 3, 3)
    currency_counts = df_clean['currency'].value_counts()
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    wedges, texts, autotexts = ax3.pie(currency_counts.values, labels=currency_counts.index, autopct='%1.1f%%',
                                       colors=colors, startangle=90, textprops={'fontweight': 'bold'})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    ax3.set_title('💱 Currency Distribution', fontweight='bold', fontsize=12)

    # 1.4 Cumulative Distribution
    ax4 = plt.subplot(2, 3, 4)
    sorted_prices = np.sort(df_ue['price_numeric'])
    cumulative = np.arange(1, len(sorted_prices) + 1) / len(sorted_prices) * 100
    ax4.plot(sorted_prices, cumulative, linewidth=2.5, color='darkblue')
    ax4.fill_between(sorted_prices, cumulative, alpha=0.3, color='skyblue')
    ax4.set_xlabel('Price (у.е.)', fontweight='bold', fontsize=11)
    ax4.set_ylabel('Cumulative %', fontweight='bold', fontsize=11)
    ax4.set_title('📈 Cumulative Distribution', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(50, color='red', linestyle=':', alpha=0.5)

    # 1.5 Price Range Categories
    ax5 = plt.subplot(2, 3, 5)
    price_ranges = pd.cut(df_ue['price_numeric'],
                          bins=[0, 300, 500, 750, 1000, 1500, 5000],
                          labels=['<300', '300-500', '500-750', '750-1K', '1K-1.5K', '>1.5K'])
    price_range_counts = price_ranges.value_counts().sort_index()
    bars = ax5.bar(range(len(price_range_counts)), price_range_counts.values, color='coral', edgecolor='black',
                   alpha=0.8)
    ax5.set_xticks(range(len(price_range_counts)))
    ax5.set_xticklabels(price_range_counts.index, rotation=45, fontsize=10)
    ax5.set_ylabel('Count', fontweight='bold', fontsize=11)
    ax5.set_title('🎯 Price Range Categories', fontweight='bold', fontsize=12)
    ax5.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2., height, f'{int(height)}', ha='center', va='bottom', fontsize=9,
                 fontweight='bold')

    # 1.6 Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    stats_text = f"""📊 PRICE STATISTICS (в у.е.)

Total Listings: {len(df_ue):,}
Mean Price: {df_ue['price_numeric'].mean():.2f}
Median Price: {df_ue['price_numeric'].median():.2f}
Std Deviation: {df_ue['price_numeric'].std():.2f}

Range: {df_ue['price_numeric'].min():.0f} - {df_ue['price_numeric'].max():.0f}
Q1 (25%): {df_ue['price_numeric'].quantile(0.25):.0f}
Q2 (50%): {df_ue['price_numeric'].quantile(0.50):.0f}
Q3 (75%): {df_ue['price_numeric'].quantile(0.75):.0f}"""
    ax6.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, pad=1))

    plt.tight_layout()
    plt.show()


# ============================================================================
# DASHBOARD 2 - DISTRICT ANALYSIS
# ============================================================================
def show_dashboard_2():
    print("\n[Opening Dashboard 2: District Analysis]")
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('TASHKENT RENTAL MARKET - DISTRICT ANALYSIS DASHBOARD',
                 fontsize=18, fontweight='bold', y=0.98)

    # 2.1 Top Districts
    ax1 = plt.subplot(2, 3, 1)
    top_districts = df_clean['district'].value_counts().head(10)
    colors_gradient = plt.cm.Spectral(np.linspace(0, 1, len(top_districts)))
    bars = ax1.barh(range(len(top_districts)), top_districts.values, color=colors_gradient, edgecolor='black',
                    alpha=0.8)
    ax1.set_yticks(range(len(top_districts)))
    ax1.set_yticklabels(top_districts.index, fontsize=10)
    ax1.set_xlabel('Number of Listings', fontweight='bold', fontsize=11)
    ax1.set_title('🏘️ Top 10 Districts', fontweight='bold', fontsize=12)
    ax1.invert_yaxis()
    for i, bar in enumerate(bars):
        ax1.text(bar.get_width(), bar.get_y() + bar.get_height() / 2., f'{int(bar.get_width())}',
                 ha='left', va='center', fontsize=9, fontweight='bold')

    # 2.2 Average Price by District
    ax2 = plt.subplot(2, 3, 2)
    district_price = df_ue.groupby('district')['price_numeric'].mean().sort_values(ascending=False).head(10)
    bars = ax2.bar(range(len(district_price)), district_price.values, color='mediumpurple', edgecolor='black',
                   alpha=0.8)
    ax2.set_xticks(range(len(district_price)))
    ax2.set_xticklabels(district_price.index, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Price (у.е.)', fontweight='bold', fontsize=11)
    ax2.set_title('💰 Avg Price by District', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), f'{int(bar.get_height())}',
                 ha='center', va='bottom', fontsize=8, fontweight='bold')

    # 2.3 Median Price by District
    ax3 = plt.subplot(2, 3, 3)
    district_median = df_ue.groupby('district')['price_numeric'].median().sort_values(ascending=False).head(10)
    bars = ax3.barh(range(len(district_median)), district_median.values, color='lightseagreen', edgecolor='black',
                    alpha=0.8)
    ax3.set_yticks(range(len(district_median)))
    ax3.set_yticklabels(district_median.index, fontsize=10)
    ax3.set_xlabel('Median Price (у.е.)', fontweight='bold', fontsize=11)
    ax3.set_title('📊 Median Price Ranking', fontweight='bold', fontsize=12)
    ax3.invert_yaxis()
    for i, bar in enumerate(bars):
        ax3.text(bar.get_width(), bar.get_y() + bar.get_height() / 2., f'{int(bar.get_width())}',
                 ha='left', va='center', fontsize=9, fontweight='bold')

    # 2.4 Price per m²
    ax4 = plt.subplot(2, 3, 4)
    district_price_m2 = df_ue_with_area.groupby('district')['price_per_m2'].mean().sort_values(ascending=False).head(10)
    bars = ax4.bar(range(len(district_price_m2)), district_price_m2.values, color='salmon', edgecolor='black',
                   alpha=0.8)
    ax4.set_xticks(range(len(district_price_m2)))
    ax4.set_xticklabels(district_price_m2.index, rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('Price/m² (у.е.)', fontweight='bold', fontsize=11)
    ax4.set_title('💎 Price per m² Value', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        ax4.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), f'{bar.get_height():.1f}',
                 ha='center', va='bottom', fontsize=8, fontweight='bold')

    # 2.5 Price Volatility (Standard Deviation)
    ax5 = plt.subplot(2, 3, 5)
    district_std = df_ue.groupby('district')['price_numeric'].std().sort_values(ascending=False).head(10)
    bars = ax5.barh(range(len(district_std)), district_std.values, color='khaki', edgecolor='black', alpha=0.8)
    ax5.set_yticks(range(len(district_std)))
    ax5.set_yticklabels(district_std.index, fontsize=10)
    ax5.set_xlabel('Standard Deviation (у.е.)', fontweight='bold', fontsize=11)
    ax5.set_title('⚡ Price Volatility', fontweight='bold', fontsize=12)
    ax5.invert_yaxis()

    # 2.6 Market Share Pie Chart
    ax6 = plt.subplot(2, 3, 6)
    top_5_districts = df_ue['district'].value_counts().head(5)
    other_count = df_ue['district'].value_counts()[5:].sum()
    pie_data = list(top_5_districts.values) + [other_count]
    pie_labels = list(top_5_districts.index) + ['Others']
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(pie_data)))
    wedges, texts, autotexts = ax6.pie(pie_data, labels=pie_labels, autopct='%1.1f%%',
                                       colors=colors_pie, startangle=90, textprops={'fontweight': 'bold'})
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    ax6.set_title('🎯 Market Share Distribution', fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.show()


# ============================================================================
# DASHBOARD 3 - ROOMS & AREA ANALYSIS
# ============================================================================
def show_dashboard_3():
    print("\n[Opening Dashboard 3: Rooms & Area Analysis]")
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('TASHKENT RENTAL MARKET - ROOMS & AREA ANALYSIS DASHBOARD',
                 fontsize=18, fontweight='bold', y=0.98)

    # 3.1 Room Distribution
    ax1 = plt.subplot(2, 3, 1)
    rooms_data = df_clean['rooms'].value_counts().sort_index().head(6)
    bars = ax1.bar(rooms_data.index.astype(str), rooms_data.values, color='steelblue', edgecolor='black', alpha=0.8)
    ax1.set_xlabel('Number of Rooms', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Count', fontweight='bold', fontsize=11)
    ax1.set_title('🏠 Listings by Room Count', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), f'{int(bar.get_height())}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 3.2 Average Price by Rooms
    ax2 = plt.subplot(2, 3, 2)
    rooms_price = df_ue.groupby('rooms')['price_numeric'].mean().sort_index().head(5)
    bars = ax2.bar(rooms_price.index.astype(str), rooms_price.values, color='coral', edgecolor='black', alpha=0.8)
    ax2.set_xlabel('Number of Rooms', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Average Price (у.е.)', fontweight='bold', fontsize=11)
    ax2.set_title('💵 Price by Room Count', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), f'{int(bar.get_height())}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 3.3 Violin Plot - Price Distribution by Rooms
    ax3 = plt.subplot(2, 3, 3)
    violin_data = [df_ue[df_ue['rooms'] == r]['price_numeric'].dropna().values
                   for r in sorted(df_ue['rooms'].dropna().unique())[:5]]
    parts = ax3.violinplot(violin_data, positions=range(len(violin_data)), showmeans=True, showmedians=True)
    ax3.set_xticks(range(len(violin_data)))
    ax3.set_xticklabels([str(int(r)) for r in sorted(df_ue['rooms'].dropna().unique())[:5]])
    ax3.set_xlabel('Number of Rooms', fontweight='bold', fontsize=11)
    ax3.set_ylabel('Price (у.е.)', fontweight='bold', fontsize=11)
    ax3.set_title('🎻 Price Distribution (Violin)', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')

    # 3.4 Area Histogram
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(df_ue['area_m2'].dropna(), bins=30, color='lightgreen', edgecolor='black', alpha=0.8)
    ax4.set_xlabel('Area (m²)', fontweight='bold', fontsize=11)
    ax4.set_ylabel('Frequency', fontweight='bold', fontsize=11)
    ax4.set_title('📐 Area Distribution', fontweight='bold', fontsize=12)
    ax4.axvline(df_ue['area_m2'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {df_ue["area_m2"].mean():.0f}')
    ax4.axvline(df_ue['area_m2'].median(), color='green', linestyle='--', linewidth=2,
                label=f'Median: {df_ue["area_m2"].median():.0f}')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)

    # 3.5 Price per m² Distribution
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(df_ue_with_area['price_per_m2'].dropna(), bins=30, color='gold', edgecolor='black', alpha=0.8)
    ax5.set_xlabel('Price per m² (у.е.)', fontweight='bold', fontsize=11)
    ax5.set_ylabel('Frequency', fontweight='bold', fontsize=11)
    ax5.set_title('💰 Price per m² Analysis', fontweight='bold', fontsize=12)
    ax5.axvline(df_ue_with_area['price_per_m2'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {df_ue_with_area["price_per_m2"].mean():.1f}')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=9)

    # 3.6 Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    stats_text = f"""📊 ROOMS & AREA STATISTICS

LISTINGS:
Total: {len(df_ue):,}
With Rooms Info: {df_ue['rooms'].notna().sum():,}
With Area Info: {df_ue['area_m2'].notna().sum():,}

ROOMS:
Most Common: {int(df_ue['rooms'].mode()[0])} room(s)

AREA (m²):
Mean: {df_ue['area_m2'].mean():.1f}
Median: {df_ue['area_m2'].median():.1f}

PRICE/m²:
Mean: {df_ue_with_area['price_per_m2'].mean():.2f}
Median: {df_ue_with_area['price_per_m2'].median():.2f}"""
    ax6.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, pad=1))

    plt.tight_layout()
    plt.show()


# ============================================================================
# DASHBOARD 4 - CORRELATION ANALYSIS
# ============================================================================
def show_dashboard_4():
    print("\n[Opening Dashboard 4: Correlation Analysis]")
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('TASHKENT RENTAL MARKET - CORRELATION & RELATIONSHIPS DASHBOARD',
                 fontsize=18, fontweight='bold', y=0.98)

    # 4.1 Scatter - Area vs Price
    ax1 = plt.subplot(2, 3, 1)
    scatter1 = ax1.scatter(df_ue_with_area['area_m2'], df_ue_with_area['price_numeric'],
                           c=df_ue_with_area['rooms'], cmap='viridis', s=50, alpha=0.6, edgecolor='black',
                           linewidth=0.5)
    ax1.set_xlabel('Area (m²)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Price (у.е.)', fontweight='bold', fontsize=11)
    ax1.set_title('📈 Price vs Area (colored by Rooms)', fontweight='bold', fontsize=12)
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Rooms', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    corr_area_price = df_ue_with_area[['area_m2', 'price_numeric']].corr().iloc[0, 1]
    ax1.text(0.05, 0.95, f'Correlation: {corr_area_price:.3f}', transform=ax1.transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    # 4.2 Scatter - Rooms vs Price
    ax2 = plt.subplot(2, 3, 2)
    df_with_rooms = df_ue[df_ue['rooms'].notna()]
    scatter2 = ax2.scatter(df_with_rooms['rooms'], df_with_rooms['price_numeric'],
                           c=df_with_rooms['area_m2'], cmap='plasma', s=50, alpha=0.6, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Number of Rooms', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Price (у.е.)', fontweight='bold', fontsize=11)
    ax2.set_title('📊 Price vs Rooms (colored by Area)', fontweight='bold', fontsize=12)
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Area (m²)', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 4.3 Heatmap - Price by Rooms & Districts
    ax3 = plt.subplot(2, 3, 3)
    pivot_data = df_ue.pivot_table(values='price_numeric', index='rooms', columns='district', aggfunc='mean')
    pivot_data = pivot_data.iloc[:5, :5]
    sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='RdYlGn_r', ax=ax3, cbar_kws={'label': 'Avg Price (у.е.)'})
    ax3.set_title('🌡️ Price by Rooms & Districts', fontweight='bold', fontsize=12)
    ax3.set_xlabel('District', fontweight='bold', fontsize=11)
    ax3.set_ylabel('Rooms', fontweight='bold', fontsize=11)

    # 4.4 Line Graph - Price Trend by Area Categories
    ax4 = plt.subplot(2, 3, 4)
    area_bins = [0, 30, 50, 70, 100, 150, 600]
    area_labels = ['0-30', '30-50', '50-70', '70-100', '100-150', '150+']
    df_ue_with_area['area_category'] = pd.cut(df_ue_with_area['area_m2'], bins=area_bins, labels=area_labels)
    area_trend = df_ue_with_area.groupby('area_category')['price_numeric'].agg(['mean', 'median', 'count'])
    ax4.plot(range(len(area_trend)), area_trend['mean'], marker='o', linewidth=2.5,
             markersize=10, label='Mean', color='darkblue')
    ax4.plot(range(len(area_trend)), area_trend['median'], marker='s', linewidth=2.5,
             markersize=10, label='Median', color='darkred')
    ax4.fill_between(range(len(area_trend)), area_trend['mean'], area_trend['median'], alpha=0.2)
    ax4.set_xticks(range(len(area_trend)))
    ax4.set_xticklabels(area_labels, fontsize=10)
    ax4.set_xlabel('Area Category (m²)', fontweight='bold', fontsize=11)
    ax4.set_ylabel('Price (у.е.)', fontweight='bold', fontsize=11)
    ax4.set_title('📉 Price Trend by Area', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)

    # 4.5 Scatter - Area vs Price per m²
    ax5 = plt.subplot(2, 3, 5)
    scatter3 = ax5.scatter(df_ue_with_area['area_m2'], df_ue_with_area['price_per_m2'],
                           c=df_ue_with_area['price_numeric'], cmap='coolwarm', s=50, alpha=0.6, edgecolor='black',
                           linewidth=0.5)
    ax5.set_xlabel('Area (m²)', fontweight='bold', fontsize=11)
    ax5.set_ylabel('Price per m² (у.е.)', fontweight='bold', fontsize=11)
    ax5.set_title('💎 Value Analysis: Area vs Price/m²', fontweight='bold', fontsize=12)
    cbar3 = plt.colorbar(scatter3, ax=ax5)
    cbar3.set_label('Total Price (у.е.)', fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # 4.6 Correlation Matrix
    ax6 = plt.subplot(2, 3, 6)
    corr_columns = df_ue_with_area[['price_numeric', 'area_m2', 'rooms']].corr()
    sns.heatmap(corr_columns, annot=True, fmt='.2f', cmap='coolwarm', ax=ax6,
                cbar_kws={'label': 'Correlation'}, vmin=-1, vmax=1, center=0)
    ax6.set_title('📊 Correlation Matrix', fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.show()


# ============================================================================
# DASHBOARD 5 - COMPARATIVE ANALYSIS
# ============================================================================
def show_dashboard_5():
    print("\n[Opening Dashboard 5: Comparative Analysis]")
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('TASHKENT RENTAL MARKET - COMPARATIVE ANALYSIS DASHBOARD',
                 fontsize=18, fontweight='bold', y=0.98)

    # 5.1 Affordable vs Expensive Districts
    ax1 = plt.subplot(2, 3, 1)
    affordable = df_ue.groupby('district')['price_numeric'].mean().sort_values().head(5)
    expensive = df_ue.groupby('district')['price_numeric'].mean().sort_values(ascending=False).head(5)
    y_pos = np.arange(len(affordable) + len(expensive))
    values = list(affordable.values) + list(expensive.values)
    labels = list(affordable.index) + list(expensive.index)
    colors_comp = ['lightgreen'] * len(affordable) + ['lightcoral'] * len(expensive)
    bars = ax1.barh(y_pos, values, color=colors_comp, edgecolor='black', alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.set_xlabel('Average Price (у.е.)', fontweight='bold', fontsize=11)
    ax1.set_title('⚖️ Affordable vs Expensive Districts', fontweight='bold', fontsize=12)
    ax1.axvline(df_ue['price_numeric'].mean(), color='blue', linestyle='--', linewidth=2, label='Market Avg')
    ax1.legend(fontsize=9)
    ax1.invert_yaxis()

    # 5.2 Listings Count Comparison
    ax2 = plt.subplot(2, 3, 2)
    rooms_count = df_ue.groupby('rooms')['price_numeric'].count().sort_index().head(5)
    x_pos = np.arange(len(rooms_count))
    bars = ax2.bar(x_pos, rooms_count.values, color=['#FF9999', '#66B2FF', '#99FF99', '#FFD700', '#FF99CC'],
                   edgecolor='black', alpha=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(rooms_count.index.astype(int), fontsize=11)
    ax2.set_xlabel('Number of Rooms', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Count', fontweight='bold', fontsize=11)
    ax2.set_title('📊 Listings by Room Type', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), f'{int(bar.get_height())}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 5.3 Stacked Bar Chart - Price Ranges by Top Districts
    ax3 = plt.subplot(2, 3, 3)
    top_5_dists = df_ue['district'].value_counts().head(5).index
    price_range_dist = []
    for dist in top_5_dists:
        dist_data = df_ue[df_ue['district'] == dist]['price_numeric']
        low = len(dist_data[dist_data < 500])
        mid = len(dist_data[(dist_data >= 500) & (dist_data < 1000)])
        high = len(dist_data[dist_data >= 1000])
        price_range_dist.append([low, mid, high])

    price_range_dist = np.array(price_range_dist)
    x = np.arange(len(top_5_dists))
    width = 0.6
    ax3.bar(x, price_range_dist[:, 0], width, label='<500', color='lightblue', edgecolor='black')
    ax3.bar(x, price_range_dist[:, 1], width, bottom=price_range_dist[:, 0],
            label='500-1K', color='lightyellow', edgecolor='black')
    ax3.bar(x, price_range_dist[:, 2], width,
            bottom=price_range_dist[:, 0] + price_range_dist[:, 1],
            label='>1K', color='lightcoral', edgecolor='black')
    ax3.set_ylabel('Number of Listings', fontweight='bold', fontsize=11)
    ax3.set_title('📦 Price Range Distribution by District', fontweight='bold', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(top_5_dists, rotation=45, ha='right', fontsize=9)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # 5.4 Room Type Distribution - Pie Chart
    ax4 = plt.subplot(2, 3, 4)
    room_availability = df_clean['rooms'].value_counts().sort_index().head(6)
    colors_room = plt.cm.Pastel1(np.linspace(0, 1, len(room_availability)))
    wedges, texts, autotexts = ax4.pie(room_availability.values,
                                       labels=[f'{int(r)}-room' for r in room_availability.index],
                                       autopct='%1.1f%%', colors=colors_room, startangle=45,
                                       textprops={'fontweight': 'bold'})
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    ax4.set_title('🏠 Room Type Distribution', fontweight='bold', fontsize=12)

    # 5.5 Average Area by Room Count
    ax5 = plt.subplot(2, 3, 5)
    avg_area_by_rooms = df_ue[df_ue['area_m2'].notna()].groupby('rooms')['area_m2'].mean().sort_index().head(5)
    bars = ax5.bar(avg_area_by_rooms.index.astype(str), avg_area_by_rooms.values,
                   color='skyblue', edgecolor='black', alpha=0.8)
    ax5.set_xlabel('Number of Rooms', fontweight='bold', fontsize=11)
    ax5.set_ylabel('Average Area (m²)', fontweight='bold', fontsize=11)
    ax5.set_title('📐 Average Area by Room Count', fontweight='bold', fontsize=12)
    ax5.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        ax5.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), f'{bar.get_height():.0f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 5.6 Comparison Summary Table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    comparison_text = f"""⚖️ MARKET COMPARISON

BY ROOM COUNT:
1-Room: {int(df_ue[df_ue['rooms'] == 1]['price_numeric'].mean())} у.е.
2-Room: {int(df_ue[df_ue['rooms'] == 2]['price_numeric'].mean())} у.е.
3-Room: {int(df_ue[df_ue['rooms'] == 3]['price_numeric'].mean())} у.е.

BY AREA:
<50 m²: {int(df_ue[df_ue['area_m2'] < 50]['price_numeric'].mean())} у.е.
50-80 m²: {int(df_ue[(df_ue['area_m2'] >= 50) & (df_ue['area_m2'] < 80)]['price_numeric'].mean())} у.е.
>80 m²: {int(df_ue[df_ue['area_m2'] > 80]['price_numeric'].mean())} у.е.

TOP 3 DISTRICTS:
Mirabad: {int(df_ue[df_ue['district'] == 'Мирабадский район']['price_numeric'].mean())} у.е.
Shaykhantahur: {int(df_ue[df_ue['district'] == 'Шайхантахурский район']['price_numeric'].mean())} у.е.
Sergeli: {int(df_ue[df_ue['district'] == 'Сергелийский район']['price_numeric'].mean())} у.е."""
    ax6.text(0.05, 0.5, comparison_text, fontsize=10, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=1))

    plt.tight_layout()
    plt.show()


# ============================================================================
# DASHBOARD 6 - MARKET INSIGHTS & TRENDS
# ============================================================================
def show_dashboard_6():
    print("\n[Opening Dashboard 6: Market Insights & Trends]")
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('TASHKENT RENTAL MARKET - INSIGHTS & TRENDS DASHBOARD',
                 fontsize=18, fontweight='bold', y=0.98)

    # 6.1 Best Value Districts
    ax1 = plt.subplot(2, 3, 1)
    best_value = df_ue_with_area.groupby('district')['price_per_m2'].mean().sort_values().head(10)
    colors_gradient = plt.cm.RdYlGn_r(np.linspace(0, 1, len(best_value)))
    bars = ax1.barh(range(len(best_value)), best_value.values, color=colors_gradient, edgecolor='black', alpha=0.8)
    ax1.set_yticks(range(len(best_value)))
    ax1.set_yticklabels(best_value.index, fontsize=10)
    ax1.set_xlabel('Price per m² (у.е.)', fontweight='bold', fontsize=11)
    ax1.set_title('💎 Best Value Districts (Lowest Price/m²)', fontweight='bold', fontsize=12)
    ax1.invert_yaxis()
    for i, bar in enumerate(bars):
        ax1.text(bar.get_width(), bar.get_y() + bar.get_height() / 2., f'{bar.get_width():.1f}',
                 ha='left', va='center', fontsize=9, fontweight='bold')

    # 6.2 Market Segmentation
    ax2 = plt.subplot(2, 3, 2)
    price_bins = [0, 250, 500, 750, 1000, 1500, 2000, 5000]
    price_labels = ['<250', '250-500', '500-750', '750-1K', '1K-1.5K', '1.5K-2K', '>2K']
    price_dist = pd.cut(df_ue['price_numeric'], bins=price_bins, labels=price_labels)
    price_dist_counts = price_dist.value_counts().sort_index()
    bars = ax2.bar(range(len(price_dist_counts)), price_dist_counts.values,
                   color=plt.cm.Spectral(np.linspace(0, 1, len(price_dist_counts))),
                   edgecolor='black', alpha=0.8)
    ax2.set_xticks(range(len(price_dist_counts)))
    ax2.set_xticklabels(price_dist_counts.index, rotation=45, fontsize=10)
    ax2.set_ylabel('Count', fontweight='bold', fontsize=11)
    ax2.set_title('🎯 Market Segmentation by Price Range', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), f'{int(bar.get_height())}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 6.3 Market Share - Top Districts Pie
    ax3 = plt.subplot(2, 3, 3)
    top_dists_pie = df_ue['district'].value_counts().head(6)
    colors_pie = plt.cm.Dark2(np.linspace(0, 1, len(top_dists_pie)))
    wedges, texts, autotexts = ax3.pie(top_dists_pie.values, labels=top_dists_pie.index, autopct='%1.1f%%',
                                       colors=colors_pie, startangle=45, textprops={'fontweight': 'bold'})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    for text in texts:
        text.set_fontsize(9)
    ax3.set_title('🏘️ Market Share - Top 6 Districts', fontweight='bold', fontsize=12)

    # 6.4 Mean vs Median Price Comparison
    ax4 = plt.subplot(2, 3, 4)
    top_dists_comp = df_ue['district'].value_counts().head(8).index
    means = [df_ue[df_ue['district'] == d]['price_numeric'].mean() for d in top_dists_comp]
    medians = [df_ue[df_ue['district'] == d]['price_numeric'].median() for d in top_dists_comp]
    x = np.arange(len(top_dists_comp))
    width = 0.35
    ax4.bar(x - width / 2, means, width, label='Mean', color='steelblue', edgecolor='black', alpha=0.8)
    ax4.bar(x + width / 2, medians, width, label='Median', color='coral', edgecolor='black', alpha=0.8)
    ax4.set_ylabel('Price (у.е.)', fontweight='bold', fontsize=11)
    ax4.set_title('📊 Mean vs Median Price by District', fontweight='bold', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(top_dists_comp, rotation=45, ha='right', fontsize=8)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    # 6.5 Room Distribution in Top Districts
    ax5 = plt.subplot(2, 3, 5)
    top_5_dists_density = df_ue['district'].value_counts().head(5).index
    room_dist_data = []
    for dist in top_5_dists_density:
        room_counts = df_ue[df_ue['district'] == dist]['rooms'].value_counts()
        room_dist_data.append(room_counts)

    room_types = [1, 2, 3, 4]
    x_pos = np.arange(len(top_5_dists_density))
    width = 0.2
    colors_rooms = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

    for i, room in enumerate(room_types):
        values = [room_dist_data[j].get(room, 0) for j in range(len(top_5_dists_density))]
        ax5.bar(x_pos + i * width, values, width, label=f'{room}-room', color=colors_rooms[i], edgecolor='black',
                alpha=0.8)

    ax5.set_ylabel('Count', fontweight='bold', fontsize=11)
    ax5.set_title('🏠 Room Type Distribution in Top 5 Districts', fontweight='bold', fontsize=12)
    ax5.set_xticks(x_pos + 1.5 * width)
    ax5.set_xticklabels(top_5_dists_density, rotation=45, ha='right', fontsize=8)
    ax5.legend(fontsize=8, loc='upper right')
    ax5.grid(True, alpha=0.3, axis='y')

    # 6.6 Key Insights Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    best_value_dist = df_ue_with_area.groupby('district')['price_per_m2'].mean().idxmin()
    most_expensive_dist = df_ue.groupby('district')['price_numeric'].mean().idxmax()
    most_affordable_dist = df_ue.groupby('district')['price_numeric'].mean().idxmin()
    most_popular_rooms = int(df_ue['rooms'].mode()[0])
    market_avg = df_ue['price_numeric'].mean()

    insights_text = f"""💡 KEY MARKET INSIGHTS

MARKET SNAPSHOT:
• Total Listings: {len(df_ue):,}
• Market Avg: {market_avg:.0f} у.е.
• Median: {df_ue['price_numeric'].median():.0f} у.е.

BEST DEALS:
• {best_value_dist}
  {df_ue_with_area[df_ue_with_area['district'] == best_value_dist]['price_per_m2'].mean():.1f} у.е./m²

MOST EXPENSIVE:
• {most_expensive_dist}
  {int(df_ue[df_ue['district'] == most_expensive_dist]['price_numeric'].mean())} у.е.

MOST AFFORDABLE:
• {most_affordable_dist}
  {int(df_ue[df_ue['district'] == most_affordable_dist]['price_numeric'].mean())} у.е.

MARKET DEMAND:
• Popular: {most_popular_rooms}-room apts
• Avg Area: {df_ue['area_m2'].mean():.0f} m²"""
    ax6.text(0.05, 0.5, insights_text, fontsize=10, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.95, pad=1))

    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN MENU
# ============================================================================
def main_menu():
    print("\n" + "=" * 80)
    print("TASHKENT RENTAL MARKET ANALYSIS - INTERACTIVE DASHBOARDS")
    print("=" * 80)
    print("\nAvailable Dashboards:")
    print("1. Price Analysis Dashboard")
    print("2. District Analysis Dashboard")
    print("3. Rooms & Area Analysis Dashboard")
    print("4. Correlation & Relationships Dashboard")
    print("5. Comparative Analysis Dashboard")
    print("6. Market Insights & Trends Dashboard")
    print("7. Exit\n")

    while True:
        try:
            choice = input("Enter your choice (1-7): ").strip()

            if choice == '1':
                show_dashboard_1()
            elif choice == '2':
                show_dashboard_2()
            elif choice == '3':
                show_dashboard_3()
            elif choice == '4':
                show_dashboard_4()
            elif choice == '5':
                show_dashboard_5()
            elif choice == '6':
                show_dashboard_6()
            elif choice == '7':
                print("\n" + "=" * 80)
                print("Thank you for using Tashkent Rental Market Analysis!")
                print("=" * 80 + "\n")
                break
            else:
                print("Invalid choice. Please try again.\n")

        except KeyboardInterrupt:
            print("\n\nExiting application...")
            break
        except Exception as e:
            print(f"Error: {e}\nPlease try again.\n")


if __name__ == "__main__":
    main_menu()