import pandas as pd
import matplotlib.pyplot as plt

def generate_plots(df_ut, df_t, df_t_coef, reg_models):
    fig, axes = plt.subplots(1, 4, figsize=(20, 12))
    axes[2].axis('off')
    axes[3].axis('off')

    # Formatting for Lasso table
    ax = axes[3]
    df_ut_table = generate_table(ax, df_ut, "Lasso")

    # Formatting for LassoCV table
    ax = axes[3]
    df_t_table = generate_table(ax, df_t, "LassoCV")

    # Formatting for LassoCV Coef_ table
    ax = axes[2]
    df_t_coef_table = generate_coef_table(ax, df_t_coef)

    # Formatting for LassoCv Coef_ plot
    ax = axes[1]
    generate_coef_plot(ax, df_t_coef)

    # Formatting for LazyRegressor Plot
    ax = axes[0]
    generate_regressor_plot(ax, reg_models)

    plt.tight_layout()
    plt.show()

def generate_table(ax, df, label):
    cell_text = [[str(val)] for val in df.iloc[0]]
    row_labels = df.columns.to_list()
    df_table = ax.table(rowLabels=row_labels, cellText=cell_text, colWidths=[.8], 
                        bbox=[0.1, 0.89, 1.2, .1], colLabels=[label], rowColours=['#dce9fa', '#bfcad9', '#dce9fa', '#bfcad9'], 
                        colColours=['#bfcad9'], cellColours=[['#dce9fa'], ['#bfcad9'], ['#dce9fa']])
    df_table.auto_set_font_size(False)
    df_table.set_fontsize(10)
    df_table.scale(1.1, 1.2)
    return df_table

def generate_coef_table(ax, df_t_coef):
    row_labels = df_t_coef['Features'].to_list()
    cell_text = [[str(val)] for val in df_t_coef['Coefficients']]
    row_colors = [None] * len(row_labels)
    colors = ['#dce9fa', '#bfcad9']
    for i, feature in enumerate(row_labels):
        row_colors[i] = colors[i % len(colors)]
    cell_colors = [[val] for val in row_colors]
    df_t_coef_table = ax.table(rowLabels=row_labels, cellText=cell_text, colWidths=[.8], 
                               loc='center', colLabels=["Coefficients"],  rowColours=row_colors, colColours=['#bfcad9','#bfcad9'], cellColours=cell_colors)
    df_t_coef_table.auto_set_font_size(False)
    df_t_coef_table.set_fontsize(10)
    df_t_coef_table.scale(1.1, 1.1)
    return df_t_coef_table

def generate_coef_plot(ax, df_t_coef):
    df_t_coef = df_t_coef.iloc[::-1]
    x_val = [val for val in df_t_coef['Coefficients']]
    y_val = df_t_coef['Features'].to_list()
    ax.barh(width=x_val ,y=y_val, height=.8, color="#bfcad9")
    ax.axvline(x=0, color='#000000', linestyle='--', linewidth=1)
    ax.set_xlabel('Coefficients')
    ax.set_ylabel('Features')

def generate_regressor_plot(ax, reg_models):
    reg_models.drop(index=['Lars'], inplace=True)
    x_val_2 = [val for val in reg_models['Adjusted R-Squared']]
    y_val_2 = reg_models.index
    ax.barh(width=x_val_2, y=y_val_2, height=.8, color="#bfcad9")
    ax.axvline(x=0, color='#000000', linestyle='--', linewidth=1)
    ax.set_xlabel('Adjusted R-Squared')
    ax.set_ylabel('Model')

if __name__ == "__main__":
    main()
