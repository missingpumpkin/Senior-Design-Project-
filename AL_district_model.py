import pandas as pd
from clean import init_df, drop_nom, mean_sub
from models import lasso_cv, lz_reg, reduce_coef, get_models

def main():
    # Load the initial dataset
    data_init = pd.read_csv('AL_Dist.csv')

    # Initialize dataframe and changeable features
    data_subset, features = init_df(data_init)

    # Subset Creation
    modify_features(features)

    # Preprocessing
    drop_nom(data_subset)
    mean_sub(data_subset)

    # Run models
    lasso_metrics, lasso_coef = lasso_cv(data_subset)
    lasso_coef_red = reduce_coef(lasso_coef)
    lzp_metrics = lz_reg(data_subset)
    new_models = get_models(lzp_metrics)

    # Display results
    display_results(data_init, data_subset, lasso_metrics, lasso_coef, lasso_coef_red, lzp_metrics, new_models, features)

def modify_features(features):
    print("Welcome Prof. Pendola, press Enter to start")
    while True:
        feature_choice = input("\nChoose feature to modify (Type 'Done' when finished): ").strip().lower()
        if feature_choice == "done":
            break
        modify_feature(features, feature_choice)

def modify_feature(features, feature_choice):
    for group in features:
        for feat in group:
            if feature_choice == feat.name.lower():
                if isinstance(feat, CategoricalFeature):
                    modify_categorical_feature(feat)
                else:
                    modify_numerical_feature(feat)

def modify_categorical_feature(feature):
    print("\nValues: ", feature.data, "\n")
    chosen_values = input("Choose variables to include (comma-separated): ").strip().split(',')
    feature.data = [value.strip() for value in chosen_values]

def modify_numerical_feature(feature):
    percentage = input("\nChoose percentage (value between 0-1): ").strip()
    feature.data = float(percentage)

def display_results(data_init, data_subset, lasso_metrics, lasso_coef, lasso_coef_red, lzp_metrics, new_models, features):
    print("\nInitial Dataset:")
    display(data_init)
    print("\nSubset Dataset:")
    display(data_subset)
    print("\nLasso Metrics:")
    display(lasso_metrics)
    print("\nLasso Coefficients:")
    display(lasso_coef)
    print("\nReduced Lasso Coefficients:")
    display(lasso_coef_red)
    print("\nLazyRegressor Metrics:")
    display(lzp_metrics)
    print("\nTop Models from LazyRegressor:")
    display(new_models)
    prt_feat_data(features)

if __name__ == "__main__":
    main()



