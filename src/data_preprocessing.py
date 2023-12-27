import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


def data_information(df):
    """
    Some important data information for further work
    :param df(Data):
    :return:
    """
    print(df.shape)
    print(df.head())
    print(df.info())
    print(df.isna().sum())
    le = LabelEncoder()
    df['geometry'] = le.fit_transform(df['geometry'])
    print(f"Correlation matrix: {df.corr()}")
    sns.pairplot(df)
    plt.suptitle('Pair Plot of Multiple Features', y=1.02)
    plt.show()


def ind_dep_columns(df):
    """
    Splitting data into independent and dependent columns
    :param df(Data):
    :return X, y:
    """
    X = df.drop(columns='chf_exp [MW/m2]')
    y = df['chf_exp [MW/m2]']
    # After correlation
    X = X.drop(columns=['D_e [mm]', 'D_h [mm]'])
    return X, y
