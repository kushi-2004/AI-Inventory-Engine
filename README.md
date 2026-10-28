# 🚀 AI-Driven Inventory & Reorder Engine

An advanced, machine-learning-powered end-to-end inventory optimization and demand forecasting web application built with Python and Streamlit. This engine utilizes LightGBM regression to project future demand, dynamically calculates Safety Stock ($SS$) and Reorder Points ($ROP$), and automates purchase order (PO) generation.

---

## ✨ Key Features

* **🤖 Machine Learning Forecasting:** Powered by a trained LightGBM regressor analyzing historical sales trends and temporal features to project 14-day future demand.
* **⚙️ Dynamic Supply Chain Calculations:** Automatically computes:
  * **Safety Stock ($SS$)** based on customizable service level targets ($Z$-scores) and demand variance.
  * **Reorder Point ($ROP$)** integrating lead-time parameters.
* **📊 Modern Glassmorphism UI:** Features a high-tech dark mode dashboard with an immersive hero viewport, background overlays, animated navigation controls, and interactive Plotly charts.
* **📥 Automated Purchase Order (PO) Generation:** Instantly flags under-stocked SKUs, generates draft purchase orders, calculates procurement costs, and provides a direct **CSV Export** option.

---

## 🛠️ Tech Stack

* **Frontend & UI:** Streamlit, Custom CSS (Glassmorphism), Plotly
* **Machine Learning & Data:** Python, Pandas, NumPy, LightGBM, Scikit-Learn
* **Version Control:** Git & GitHub

---

## 📂 Project Structure

```text
AI-Inventory-Engine/
│
├── data/                  # Staging datasets & historical sales logs
├── app.py                 # Main Streamlit application script
├── requirements.txt       # Python package dependencies
└── README.md              # Project documentation
