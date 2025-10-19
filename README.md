# 📊 Lulu UAE Advanced Analytics Dashboard

A powerful Streamlit-based analytics platform for comprehensive sales analysis, customer segmentation, and ad campaign performance tracking for Lulu UAE hypermarkets.

## 🎯 Overview

This dashboard provides multi-dimensional insights into:
- **Customer Behavior** by age groups and demographics
- **Ad Campaign Performance** across different cities and store formats
- **Category Targeting Strategies** with AI-powered recommendations
- **Sales Trends** and revenue optimization opportunities
- **Promotional Effectiveness** and ROI analysis

---

## ✨ Key Features

### 📊 Overview Dashboard
- Real-time KPI metrics (Total Sales, Transactions, Avg Basket, Quantity)
- Sales breakdown by city with interactive charts
- Performance analysis by store type

### 👥 Customer Personas
- 7 age group segments: 13-17, 18-24, 25-34, 35-44, 45-54, 55-64, 65+
- Category preferences by age group
- Heatmap showing age-category relationships
- Demographic insights with persona statistics

### 📢 Ad Campaign Analysis
- Campaign performance rankings with ROI scoring
- City-wise campaign effectiveness
- Promo engagement rates and success indicators
- Why campaigns work or fail analysis

### 🎯 Category Targeting Strategy
- Top 15 product categories by sales
- Deep-dive analysis for any category
- Age group & campaign performance per category
- AI-generated targeting recommendations with action plans

### 📈 Advanced Comparisons
- Age Group × Campaign heatmap
- Campaign × Category analysis
- City × Age Group geographic insights
- Top 10 performing combinations
- Multi-dimensional data export

---

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Step 1: Install Dependencies
```bash
pip install streamlit pandas numpy altair
```

### Step 2: Prepare Your Data
Place your CSV file at `/mnt/data/lulu_uae_master_2000.csv` or use the upload feature in the dashboard.

### Step 3: Run the Dashboard
```bash
streamlit run lulu_advanced_analytics_dashboard.py
```

Dashboard opens at `http://localhost:8501`

---

## 📁 Required CSV Format

Your CSV should include these columns (auto-detected):

| Column | Type | Example Values |
|--------|------|---|
| SalesAmount | Numeric | 1250.50 |
| Quantity | Numeric | 3 |
| Department | Text | Groceries |
| Store_format | Text | Hypermarket, Express |
| Category | Text | Dairy, Electronics |
| Product | Text | Milk-1L, Phone X |
| Campaign | Text | Summer Sale, Tech Week |
| PromoCode | Text | PROMO20, DISC10 |
| Gender | Text | M, F |
| Age | Numeric | 25, 35, 45 |
| Nationality | Text | UAE, Indian, Filipino |
| City | Text | Dubai, Abu Dhabi |
| Transaction | Text | TXN001, TXN002 |
| Date | Date | 2024-01-15 |

### Sample CSV
```
SalesAmount,Quantity,Department,Store_format,Category,Product,Campaign,PromoCode,Gender,Age,Nationality,City,Transaction,Date
1250.50,3,Groceries,Hypermarket,Dairy,Milk-1L,Summer Sale,PROMO20,M,35,UAE,Dubai,TXN001,2024-01-15
850.00,2,Electronics,Express,Mobile,Phone X,New Year,DISC10,F,28,Indian,Abu Dhabi,TXN002,2024-01-16
```

---

## 📖 How to Use

### Global Filters (Left Sidebar)
- **🏙️ City**: Filter by specific city
- **🏬 Store Type**: Filter by store format
- **📅 Date Range**: Select date period

### Navigation Pages

#### 📊 Overview
- View total sales, transactions, avg basket size
- See sales breakdown by city and store type
- Interactive charts with hover tooltips

#### 👥 Customer Personas
1. View age distribution
2. Select age group from dropdown
3. See top categories for that age group
4. Check heatmap for age × category relationships
5. Review persona statistics

#### 📢 Ad Campaign Analysis
1. View all campaign rankings
2. Select campaign to analyze by city
3. Check engagement rates and ROI
4. See performance indicators (✅ High / ⚠️ Medium / ❌ Low)

#### 🎯 Category Targeting Strategy
1. Browse top 15 categories
2. Select category for deep dive
3. View best age group and campaign for category
4. Read AI-generated recommendations
5. Implement action plan

#### 📈 Advanced Comparisons
1. View age × campaign heatmap
2. Check campaign × category relationships
3. Analyze city × age group performance
4. Export multi-dimensional analysis as CSV

### Chart Interactions
- **Hover**: See detailed information
- **Zoom**: Drag to select area
- **Pan**: Hold shift and drag
- **Legend**: Click to toggle data

---

## 🔧 Features Explained

### Age Groups
- **13-17**: Gen Z (Teens)
- **18-24**: Gen Z (Young Adults)
- **25-34**: Millennials
- **35-44**: Gen X (Early)
- **45-54**: Gen X (Late)
- **55-64**: Baby Boomers
- **65+**: Silent Generation

### Campaign Performance Levels
- **✅ High Engagement** (>30% promo): Campaign messaging resonates well
- **⚠️ Medium Engagement** (15-30%): Has potential, needs optimization
- **❌ Low Engagement** (<15%): Needs revision or discontinuation

### ROI Score
```
ROI Score = Total Campaign Sales / Transaction Count
(Higher = Better revenue per transaction)
```

---

## 🚀 Deployment

### Streamlit Cloud
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" → Select repository & file
4. Deploy!

### Docker
```bash
docker build -t lulu-dashboard .
docker run -p 8501:8501 lulu-dashboard
```

### Local Server
```bash
streamlit run lulu_advanced_analytics_dashboard.py \
  --server.port 8080 \
  --server.address 0.0.0.0
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'streamlit'"
```bash
pip install streamlit pandas numpy altair
```

### "Could not load CSV at /mnt/data/..."
- Use upload feature in sidebar
- Ensure CSV format matches requirements
- Check file exists at specified path

### Charts not displaying
- Refresh browser page
- Clear browser cache
- Ensure data is properly formatted
- Check for null values in key columns

### Dashboard runs slow
- Filter by date range first
- Use city/store filters to reduce rows
- Increase data load time in settings

### Age groups showing "Unknown"
- Ensure Age column has numeric values
- Check for missing or text values
- Dashboard auto-generates random ages if missing

---

## 📊 Example Analysis: Increase Electronics Sales to 18-24 Age Group

1. **Customer Personas** → Select "18-24" → See Electronics is top category
2. **Category Targeting** → Select "Electronics" → Check best campaign (e.g., "Tech Sale")
3. **Ad Campaign Analysis** → Check "Tech Sale" performance in Dubai
4. **Result** → Dashboard recommends targeting 18-24 with "Tech Sale" campaign
5. **Export** → Download analysis → Share with marketing team → Implement

---

## 📁 Project Structure

```
lulu-analytics-dashboard/
├── lulu_advanced_analytics_dashboard.py  # Main dashboard script
├── requirements.txt                       # Python dependencies
├── README.md                              # This file
└── sample_data.csv                        # Sample data file
```

---

## 📋 Requirements (requirements.txt)

```
streamlit==1.28.1
pandas==2.0.3
numpy==1.24.3
altair==5.0.1
```

---

## 🎯 Performance Tips

- Filter data before analysis using sidebar
- Use date range to limit dataset
- Select specific city/store for faster loading
- Export CSV for external analysis
- Cache results using Streamlit cache

---

## 📞 Support

- Check troubleshooting section above
- Review data format requirements
- Verify CSV columns are properly named
- Test with sample data first
- Check Streamlit logs for errors

---

## 📈 Metrics & KPIs

- **Total Sales**: Sum of all transactions in filtered period
- **Transactions**: Count of unique transaction IDs
- **Avg Basket**: Average sales per transaction
- **Total Quantity**: Total items purchased
- **Promo Rate**: % of transactions using promo codes
- **ROI Score**: Sales per transaction by campaign

---

## 🔐 Data Privacy

- No data is stored externally
- CSV file remains on your local/server
- Dashboard processes data in-memory only
- Export downloads to your device
- Clear browser cache for privacy

---

## ✅ Version

**Version 1.0** - January 2025  
**Status**: Production Ready  
**Last Updated**: January 2025

---

## 📚 Resources

- [Streamlit Docs](https://docs.streamlit.io)
- [Altair Gallery](https://altair-viz.github.io/gallery/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

---

**Built for Lulu UAE Analytics Team** | All Rights Reserved
