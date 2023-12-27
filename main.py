from src import data, data_preprocessing, training_the_model, model_accuracy


def main():
    file_path = 'data/Data_CHF_Zhao_2020_ATE.csv'

    # Importing Data
    df = data.import_data(file_path)

    # Get Data information and Split the columns
    data_preprocessing.data_information(df)
    X, y = data_preprocessing.ind_dep_columns(df)

    # Train the model
    y_test, y_pred = training_the_model.train_model(X, y)

    # Get model accuracies
    model_accuracy.accuracies(y_test, y_pred)


if __name__ == "__main__":
    main()