# 📊 Smart Data Analyzer

An advanced Streamlit-based web app that lets users upload datasets (CSV/XLSX) and perform **data cleaning**, **visualization**, **AI-powered insights**, **natural language queries**, **machine learning predictions**, and **dashboard generation** — all in one place.

---

## 🚀 Features

### ✅ Data Upload and Preview
- Upload `.csv` or `.xlsx` files
- Preview top rows, column names, and dataset shape

### 📌 Data Summary
- Full statistical summary (`describe()`)
- Missing values overview
- Column data types and null counts

### 📊 Interactive Visualizations
- Filter dataset by column values
- Generate Bar, Line, Scatter, Box, and Pie charts using Plotly
- Group-by options for aggregated views

### 🧹 Data Cleaning Tools
- Fill missing values using Mean / Median / Mode
- Drop null rows and duplicate records
- Convert column data types
- Download cleaned dataset

### 🧠 AI-Powered Insights (OpenAI GPT)
- Auto-generated prompts analyze top 30 rows
- Get natural language summaries and observations

### 💬 Ask Your Data (Natural Language Queries)
- Ask questions like: _"Top 5 job titles by salary?"_
- AI returns executable Pandas code
- Executes safely and displays result

### 📈 ML Prediction (No-Code Model Builder)
- Select features and target column
- Choose classification or regression
- Train a RandomForest model
- Displays:
  - Accuracy, Precision, Recall, F1-score
  - R², MAE, RMSE (for regression)
  - Confusion matrix, actual vs predicted plot
  - Z-score outlier detection
- Download trained model (.pkl)

### 📊 Dashboard View
- Dataset overview (rows, columns, missing values)
- Top categories in selected column
- Correlation heatmap for numeric data

### 📄 PDF Report Export
- Download a PDF report including:
  - Dataset info
  - Column list
  - AI insights
  - Last generated chart (as image)

---

## 🛠️ Tech Stack

- **Frontend/UI:** Streamlit
- **Backend/Logic:** Python
- **AI:** OpenAI GPT-3.5 Turbo (via API)
- **Visualization:** Plotly, Matplotlib, Seaborn
- **ML:** Scikit-learn, Joblib
- **PDF:** fpdf
- **Other:** pandas, numpy, python-dotenv

---

## 🧪 How to Run

### 1. Clone this repo

```bash
git clone https://github.com/YOUR_USERNAME/smart-data-analyzer.git
cd smart-data-analyzer

### 2. Create and activate virtual environment (optional)
python -m venv venv

source venv/bin/activate   # or venv\Scripts\activate (on Windows)
 ### 3. Install requirements
pip install -r requirements.txt

### 4. Add your .env file
OPENAI_API_KEY=your_openai_key_here

### 5. Run the app
streamlit run app.py

📦 Folder Structure
smart-data-analyzer/
├── app.py
├── ml_model.py
├── dashboard.py
├── .env
├── .gitignore
├── requirements.txt
├── utils/
│   └── cleaning.py
├── components/
│   └── insights.py



📤 Deployment Options
      Streamlit Cloud
      Render
      Hugging Face Spaces (with Gradio frontend)

🙌 Acknowledgements
Streamlit
OpenAI
Scikit-learn
Plotly
