import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

def lasso_cv(df):
    # Set target and data
    X = df.drop('achvz', axis=1)
    y = df['achvz']

    # train_test_split: test=.2, train=.8
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=64)

    # Setup Lasso
    lasso = Lasso()

    # Lasso cross validation with tuning
    param_grid = {
        'alpha': [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        'tol': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    }
    lasso_cv = GridSearchCV(lasso, param_grid, cv=3, n_jobs=-1)
    lasso_cv.fit(X_train, y_train)
    y_pred2 = lasso_cv.predict(X_test)
    
    # Results
    lasso2 = lasso_cv.best_estimator_
    lasso2.fit(X_train, y_train)

    df_t = pd.DataFrame(columns=["Mean absolute Error", "Mean Squared Error", "R2 Score", "Lasso Vars"])
    values = [mean_absolute_error(y_test, y_pred2), mean_squared_error(y_test, y_pred2), r2_score(y_test, y_pred2), lasso_cv.best_estimator_]
    df_t.loc[0] = values

    feature_names = df.columns.tolist()
    feature_names.remove('achvz')

    df_t_coef = pd.DataFrame(lasso2.coef_, columns=['Coefficients'], index=feature_names)
    df_t_coef_sorted = df_t_coef.iloc[np.argsort(np.abs(df_t_coef['Coefficients']))]
    
    return df_t, df_t_coef_sorted

def lz_reg(df):
    # Set target and data
    X = df.drop('achvz', axis=1)
    y = df['achvz']

    # train_test_split: test=.2, train=.8
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=64)

    reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
    reg_models, reg_predictions = reg.fit(X_train, X_test, y_train, y_test)

    return reg_models

def get_models(df, amount=3):
    temp_df = df.copy()
    temp_df = temp_df.head(amount)
    return temp_df

def reduce_coef(df, reduc=0.05):
    temp_df = df.copy()
    temp_df = temp_df[abs(temp_df['Coefficients']) >= reduc]
    return temp_df
