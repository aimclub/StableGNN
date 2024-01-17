import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from stable_gnn.fairness import Fair


def print_results(res):
    """Print results of fairness improvement algorithm."""
    accuracy_absolute_loss = res["accuracy_of_initial_classifier"] - res["accuracy_of_fair_classifier"]
    accuracy_percentage_loss = accuracy_absolute_loss / res["accuracy_of_initial_classifier"] * 100
    fairness_absolute_improvement = res["fairness_of_initial_classifier_diff"] - res["fairness_of_fair_classifier_diff"]
    fairness_percentage_improvement = fairness_absolute_improvement / res["fairness_of_initial_classifier_diff"] * 100

    f_accuracy = f"Accuracy of initial classifier is {res['accuracy_of_initial_classifier']:0.4f}, while accuracy of fair classifier is \
{res['accuracy_of_fair_classifier']:0.4f}. Accuracy loss is {accuracy_absolute_loss:0.4f}; it has decreased on \
{accuracy_percentage_loss:0.4f}%."

    f_fairness = f"Cuae-difference of initial classifier is {res['fairness_of_initial_classifier_diff']:0.4f}, while cuae-difference of fair \
classifier is {res['fairness_of_fair_classifier_diff']:0.4f}. Fairness improvement is {fairness_absolute_improvement:0.4f}; it has \
increased on {fairness_percentage_improvement:0.4f}%. "

    print("")
    print(f_accuracy)
    print("")
    print(f_fairness)
    return {
        "fair_accuracy": res["accuracy_of_fair_classifier"],
        "initial_accuracy": res["accuracy_of_initial_classifier"],
        "fair_fairness": res["fairness_of_fair_classifier_diff"],
        "initial_fairness": res["fairness_of_initial_classifier_diff"],
    }


def run(name, init_cl):
    """Run experiment."""
    initial_classifier = init_cl
    if name == "LOAN":
        loan = pd.read_csv("loan_cleaned.csv")
        loan = loan[loan["loan_status"] != "Current"]

        # create risk groups 0 - good, 1 - bad, 2 - dubious
        def loan_grouper(x):
            if x == "Fully Paid":
                z = 0
            elif x == "Charged Off":
                z = 1
            elif x == "Late (31-120 days)":
                z = 2
            elif x == "Issued":
                z = 2
            elif x == "In Grace Period":
                z = 2
            elif x == "Late (16-30 days)":
                z = 2
            elif x == "Does not meet the credit policy. Status:Fully Paid":
                z = 2
            elif x == "Default":
                z = 1
            elif x == "Does not meet the credit policy. Status:Charged Off":
                z = 1
            return z

        loan["target"] = loan["loan_status"].apply(loan_grouper)
        loan = loan[
            [
                "loan_amnt",
                "term",
                "int_rate",
                "verification_status",
                "initial_list_status",
                "target",
                "sub_grade",
                "home_ownership",
                "purpose",
                "dti",
                "revol_bal",
                "total_pymnt",
                "total_rec_prncp",
            ]
        ]
        loan = pd.get_dummies(loan, drop_first=True)
        loan = loan.rename(columns={"initial_list_status_w": "attr"})

        fairness = Fair(dataset=loan, estimator=initial_classifier)

    res = fairness.run(number_iterations=30, interior_classifier="rf", multiplier=20)

    return res


if __name__ == "__main__":
    accs = []
    fairs = []
    for i in range(10):
        dic = run(name="LOAN", init_cl=MLPClassifier(max_iter=300))
        accs.append(dic["accuracy_of_fair_classifier"])
        fairs.append(dic["fairness_of_fair_classifier_diff"])
    print(np.mean(accs), np.std(accs), np.mean(fairs), np.std(fairs))
