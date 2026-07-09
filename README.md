# 🎓 Placement Package Predictor

A machine learning project that predicts a student's placement package (in Lacs) based on their CGPA using **Simple Linear Regression**.

The trained model is deployed as both a **Flask web app** and a **Streamlit app**.

---

## 📊 Dataset

| Column    | Description                          |
|-----------|--------------------------------------|
| `cgpa`    | Student's CGPA (range: 4.26 – 9.58) |
| `package` | Placement package in Lacs (target)   |

- **200** student records
- Source: uploaded via Google Colab

---

## 🧠 Model

| Property       | Value                        |
|----------------|------------------------------|
| Algorithm      | Simple Linear Regression     |
| Library        | `scikit-learn`               |
| Input feature  | `cgpa`                       |
| Output         | `package` (in Lacs)          |
| Slope (m)      | ≈ 0.558                      |
| Intercept (c)  | ≈ −0.896                     |
| RMSE (test)    | ≈ 0.348                      |
| MAE (test)     | ≈ 0.288                      |
| R² (test)      | ≈ 0.78                       |
| Serialized as  | `placement_lr.pkl` (joblib)  |

The model follows: **Package = 0.558 × CGPA − 0.896**

---

## 📁 Project Structure

```
placement-linear-regression/
│
├── placement_lr.pkl                        # Serialized trained model (joblib)
│
├── app.py                                  # Flask web app
├── streamlit_app.py                        # Streamlit app (Streamlit Cloud ready)
├── requirements.txt                        # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

---

### ▶️ Run the Flask App

```bash
python app.py
```

Open your browser at **http://127.0.0.1:5000**

**Endpoints:**

| Method | Route      | Description                        |
|--------|------------|------------------------------------|
| GET    | `/`        | Renders the prediction form        |
| POST   | `/predict` | Returns the predicted package      |

**REST API example (JSON):**

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"cgpa": 8.0}'
```

**Response:**
```json
{
  "cgpa": 8.0,
  "predicted_package_lacs": 3.57
}
```

---

### ▶️ Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

Open your browser at **http://localhost:8501**

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repository (including `placement_lr.pkl`) to **GitHub**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Select your repository and branch.
4. Set **Main file path** to `streamlit_app.py`.
5. Click **Deploy** — your app will be live in seconds! 🎉

---

## 📸 App Preview

| Flask App | Streamlit App |
|-----------|---------------|
| Dark glassmorphism UI | Interactive CGPA slider |
| Form-based + JSON API | One-click prediction |
| Runs locally | Deployable on Streamlit Cloud |

---

## 🛠 Tech Stack

- **Python 3.x**
- **scikit-learn** — Linear Regression model
- **joblib** — Model serialization
- **Flask** — REST API & web interface
- **Streamlit** — Interactive data app
- **NumPy** — Numerical computation

---

## 📓 Notebook Walkthrough

The Jupyter notebook [link]( https://colab.research.google.com/drive/1hsIQFMc_Eo1Sg136t5n0vnx66HIwHq2q?usp=sharing) covers:

1. **Data Loading** — Import CSV via Google Colab
2. **EDA** — Scatter plot of CGPA vs Package
3. **Preprocessing** — Train/test split (80/20, `random_state=2`)
4. **Model Training** — `LinearRegression().fit(X_train, Y_train)`
5. **Evaluation** — RMSE, MAE, R² on test set
6. **Model Export** — `joblib.dump(lr, "placement_lr.pkl")`

---

## 📄 License

This project is open-source and free to use for educational purposes.
