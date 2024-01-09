import pandas as pd


# auxiliary functions for data preparing
def grouper_race(x):
    if x == 7:
        return 1
    else:
        return 0


def grouper_gpa(x):
    if x > 3.4:
        return 2
    elif x < 3.1:
        return 0
    else:
        return 1


def prepare_data(name="bar_pass_prediction"):
    if name == "bar_pass_prediction":
        d = pd.read_csv("bar_pass_prediction.csv")
        for x in [
            "ID",
            "race1",
            "race2",
            "sex",
            "bar",
            "dnn_bar_pass_prediction",
            "pass_bar",
            "indxgrp2",
            "gender",
            "grad",
            "Dropout",
            "fulltime",
            "lsat",
            "zfygpa",
            "ugpa",
            "zgpa",
            "other",
            "asian",
            "black",
            "hisp"
        ]:
            del d[x]

        d["gpa"] = d["gpa"].apply(grouper_gpa)
        d["race"] = d["race"].apply(grouper_race)
        d = d.rename(columns={"gpa": "target", "race": "attr"})
        d = pd.get_dummies(d, drop_first=True)
        d = d.dropna(how="any")

        return d
