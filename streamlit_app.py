import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

st.set_page_config(page_title="Churn Predictor", layout="wide")

#Sidebar 
page = st.sidebar.radio("Menu", ["EDA", "Train Model", "Predict"])

# load data
DATA_PATH = r"D:\Telecom Customer Churn Project\dataset\Telco-Customer-Churn.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.drop_duplicates(inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    for col in df.select_dtypes(include='object').columns:
        if col != 'customerID':
            df[col] = df[col].str.strip().str.capitalize()
    return df

df = load_data()

# session state
if 'trained' not in st.session_state: st.session_state['trained'] = False


# Page_1 — (EDA)
if page == "EDA":
    st.title("Exploratory Data Analysis")
    st.write(f"Dataset shape: **{df.shape[0]:,} rows × {df.shape[1]} columns**")
    st.dataframe(df.head())
    st.markdown("---")

    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Churn', data=df, palette='magma', ax=ax)
    st.pyplot(fig); plt.close()

    st.subheader("Churn by Contract Type")
    fig, ax = plt.subplots()
    sns.countplot(x='Contract', hue='Churn', data=df, palette='viridis', ax=ax)
    st.pyplot(fig); plt.close()

    st.subheader("Monthly Charges by Churn")
    fig, ax = plt.subplots()
    sns.kdeplot(df[df['Churn'] == 'No']['MonthlyCharges'],  label='Stayed',  fill=True, ax=ax)
    sns.kdeplot(df[df['Churn'] == 'Yes']['MonthlyCharges'], label='Churned', fill=True, ax=ax)
    ax.legend(); st.pyplot(fig); plt.close()

# Page_2 (Train)
elif page == "Train Model":
    st.title("Train Stacking Model")

    if st.button("Start Training"):
        data = df.copy()
        if 'customerID' in data.columns: data.drop('customerID', axis=1, inplace=True)
        data['Churn'] = data['Churn'].replace({'Yes': 1, 'No': 0})
        data = pd.get_dummies(data, drop_first=True)

        X, y = data.drop('Churn', axis=1), data['Churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        imputer = SimpleImputer(strategy='median')
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
        X_test  = pd.DataFrame(imputer.transform(X_test),      columns=X.columns)

        scaler = StandardScaler()
        num = ['tenure', 'MonthlyCharges', 'TotalCharges']
        X_train[num] = scaler.fit_transform(X_train[num])
        X_test[num]  = scaler.transform(X_test[num])

        base = [('lr', LogisticRegression(max_iter=1000, random_state=42)),
                ('dt', DecisionTreeClassifier(random_state=42)),
                ('rf', RandomForestClassifier(random_state=42)),
                ('gb', GradientBoostingClassifier(random_state=42))]
        model = StackingClassifier(estimators=base, final_estimator=LogisticRegression(), cv=5)

        with st.spinner("Training in progress..."):
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        st.session_state['model']        = model
        st.session_state['scaler']       = scaler
        st.session_state['imputer']      = imputer
        st.session_state['feature_cols'] = X.columns.tolist()
        st.session_state['trained']      = True

        st.success("Training complete!")
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
        col2.metric("ROC-AUC",  f"{roc_auc_score(y_test, y_prob):.4f}")

    elif st.session_state['trained']:
        st.info("Model already trained. Go to Predict page.")

# page_3  (Podict)
elif page == "Predict":
    st.title("Predict Customer Churn")

    if not st.session_state['trained']:
        st.warning("Please train the model first."); st.stop()

    tenure   = st.slider("Tenure (months)", 0, 72, 12)
    monthly  = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
    total    = st.number_input("Total Charges ($)",   0.0, 10000.0, 800.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["Fiber optic", "Dsl", "No"])

    if st.button("Predict"):
        row = {'tenure': tenure, 'MonthlyCharges': monthly, 'TotalCharges': total,
               'Contract': contract, 'InternetService': internet}

        inp = pd.get_dummies(pd.DataFrame([row]))
        for c in st.session_state['feature_cols']:
            if c not in inp: inp[c] = 0
        inp = inp[st.session_state['feature_cols']]

        inp = pd.DataFrame(st.session_state['imputer'].transform(inp), columns=inp.columns)
        inp[['tenure', 'MonthlyCharges', 'TotalCharges']] = st.session_state['scaler'].transform(
            inp[['tenure', 'MonthlyCharges', 'TotalCharges']])

        pred = st.session_state['model'].predict(inp)[0]
        prob = st.session_state['model'].predict_proba(inp)[0][1]

        if pred == 1:
            st.error(f"High Churn Risk — Probability: {prob*100:.1f}%")
        else:
            st.success(f"Likely to Stay — Churn Probability: {prob*100:.1f}%")

        st.progress(float(prob))