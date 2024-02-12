import pandas as pd
from clean import preproc, init_df, drop_nom, mean_sub

def main():
    # Load the initial dataset
    data_init = pd.read_csv('AL_Dist.csv')

    # Preprocessing
    data_subset, features = init_df(data_init)
    drop_nom(data_subset)
    mean_sub(data_subset)

    # Display cleaned dataset and features
    display(data_subset)
    prt_feat_data(features)

if __name__ == "__main__":
    main()
