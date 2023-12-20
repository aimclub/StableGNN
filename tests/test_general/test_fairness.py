import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from stable_gnn.fairness import Fair


def synthetic_dataset(size=1000, influence=True):
    """
    param: size (int) Number of observations
    param: influence (bool) Flag if True, then dependence between sensitive attribute and label is imposed. If False, then the
                   sensitive attribute is independent of the label.
    return: synthetic_df (pd.DataFrame) Synthetic dataset
    """

    attr = np.random.choice([0, 1], size=size)
    error_x = np.random.normal(loc=0.0, scale=0.3, size=size)
    error_y = np.random.normal(loc=0.0, scale=0.3, size=size)
    error_z = np.random.normal(loc=0.0, scale=0.3, size=size)
    error_target = np.random.normal(loc=0.0, scale=0.5, size=size)

    y1 = np.random.normal(loc=1, scale=1, size=size)
    y2 = np.random.normal(loc=1, scale=1, size=size)
    y3 = np.random.normal(loc=1, scale=1, size=size)

    x = y1 + y2 + error_x
    y = y1 + y3 + error_y
    z = y2 + y3 + error_z

    if influence == True:
        target = x * (1 + 2 * attr) + y * (1 - 0.5 * attr) + z * (1 + 0.5 * attr) + error_target * attr
    if influence == False:
        target = x + y + z + error_target

    target = simple_splitter(target)

    synthetic_df = pd.DataFrame(np.array((x, y, z, attr, target))).T.rename(
        columns={0: "x", 1: "y", 2: "z", 3: "attr", 4: "target"}
    )

    return synthetic_df


def simple_splitter(arr):
    arr_unchanged = arr.copy()
    arr = np.sort(np.array(arr))
    l = len(arr)
    n1 = arr[int(l / 3)]
    n2 = arr[int(2 * l / 3)]
    result = []

    for i in range(l):
        if arr_unchanged[i] <= n1:
            result.append(0)
        elif arr_unchanged[i] > n1 and arr_unchanged[i] <= n2:
            result.append(1)
        else:
            result.append(2)

    return np.array(result)


def test_fairness():
    d = synthetic_dataset(400)
    dataset = synthetic_dataset(20000)
    random_state = 78
    cl = LogisticRegression(random_state=random_state)
    y = d.drop("target", axis=1)
    x = d["target"]
    y_train, y_test, x_train, x_test = train_test_split(y, x, random_state=random_state)
    cl.fit(y_train, x_train)
    fairness = Fair(dataset, estimator=cl)
    accs_fair = []
    accs_init = []
    fairs_fair = []
    fairs_init = []
    for _ in range(15):
        res = fairness.run(
            number_iterations=10,
            prefit=True,
            interior_classifier="knn",
            verbose=True,
            multiplier=30,
            random_state=random_state,
        )
        accs_fair.append(res["accuracy_of_fair_classifier"])
        accs_init.append(res["accuracy_of_initial_classifier"])
        fairs_fair.append(res["fairness_of_fair_classifier_diff"])
        fairs_init.append(res["fairness_of_initial_classifier_diff"])
    assert (np.mean(accs_init) - np.mean(accs_fair)) <= 0.05
    assert np.mean(fairs_fair) <= np.mean(fairs_init)
