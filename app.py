import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

df_1 = pd.read_csv(
    r"C:\customer_churn_analysis\anaconda_projects\ea39725e-e0be-4238-a44f-a521c44e9093\tel_churn.csv"
)

model = pickle.load(open("model.sav", "rb"))


@app.route("/")
def loadPage():
    return render_template("home.html")


@app.route("/", methods=["POST"])
def predict():

    SeniorCitizen = int(request.form["query1"])
    MonthlyCharges = float(request.form["query2"])
    TotalCharges = float(request.form["query3"])
    tenure = int(request.form["query19"])

    gender = request.form["query4"]
    Partner = request.form["query5"]
    Dependents = request.form["query6"]
    PhoneService = request.form["query7"]
    MultipleLines = request.form["query8"]
    InternetService = request.form["query9"]
    OnlineSecurity = request.form["query10"]
    OnlineBackup = request.form["query11"]
    DeviceProtection = request.form["query12"]
    TechSupport = request.form["query13"]
    StreamingTV = request.form["query14"]
    StreamingMovies = request.form["query15"]
    Contract = request.form["query16"]
    PaperlessBilling = request.form["query17"]
    PaymentMethod = request.form["query18"]

    data = [[SeniorCitizen, MonthlyCharges, TotalCharges, gender,
             Partner, Dependents, PhoneService, MultipleLines,
             InternetService, OnlineSecurity, OnlineBackup,
             DeviceProtection, TechSupport, StreamingTV,
             StreamingMovies, Contract, PaperlessBilling,
             PaymentMethod, tenure]]

    new_df = pd.DataFrame(data, columns=[
        "SeniorCitizen", "MonthlyCharges", "TotalCharges", "gender",
        "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "tenure"
    ])

    df_2 = pd.concat([df_1, new_df], ignore_index=True)

    df_2["tenure"] = pd.to_numeric(df_2["tenure"], errors="coerce").fillna(0)
    df_2["TotalCharges"] = pd.to_numeric(df_2["TotalCharges"], errors="coerce").fillna(0)

    labels = [f"{i} - {i+11}" for i in range(1, 72, 12)]
    df_2["tenure_group"] = pd.cut(
        df_2["tenure"],
        bins=range(1, 80, 12),
        right=False,
        labels=labels
    )

    df_2.drop(columns=["tenure"], inplace=True)

    final_df = pd.get_dummies(df_2[[
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod",
        "tenure_group"
    ]])

    final_input = final_df.tail(1)

    # Align columns to training features
    final_input = final_input.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input)[0][1]

    if prediction == 1:
        output1 = "Customer likely to churn."
    else:
        output1 = "Customer likely to stay."

    output2 = f"Confidence: {probability * 100:.2f}%"

    return render_template("home.html", output1=output1, output2=output2)


if __name__ == "__main__":
    app.run(debug=True)
