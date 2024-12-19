from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from preparation import read_data


def recursive_elimination(scoring="neg_mean_squared_error"):
    # Load and clean data
    x, y = read_data()

    # data normalisation
    x = (x-x.mean())/x.std()

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    min_features_to_select = 1  # Minimum number of features to consider
    clf = LinearRegression()
    cv = StratifiedKFold(5)

    rfecv = RFECV(
        estimator=clf,
        step=1,
        cv=cv,
        scoring=scoring,
        min_features_to_select=min_features_to_select,
        n_jobs=2,
    )
    rfecv.fit(x_train, y_train.to_numpy().flatten())

    print(f"Optimal number of features: {rfecv.n_features_}")