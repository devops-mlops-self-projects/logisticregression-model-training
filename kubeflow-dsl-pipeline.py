from kfp import dsl
from kfp import kubernetes

@dsl.component(base_image='kartikjena33/kfp:1.0.5')
def ingest_data() -> str:
    import pandas as pd
    import os

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    data = pd.read_csv(url, sep=';')
    file_path = '/data/winequality.csv'
    data.to_csv(file_path, index=False)
    
    # Log file creation
    if os.path.exists(file_path):
        print(f"File created successfully: {file_path}")
    else:
        print(f"Failed to create file: {file_path}")
    
    return file_path

@dsl.component(base_image='kartikjena33/kfp:1.0.5')
def preprocess_data(input_path: str) -> str:
    import pandas as pd
    import os

    # Log file path
    print(f"Input file path: {input_path}")
    
    if os.path.exists(input_path):
        data = pd.read_csv(input_path)
        
        # Mean imputation
        data.fillna(data.mean(), inplace=True)
        
        output_path = '/data/preprocessed_winequality.csv'
        data.to_csv(output_path, index=False)
        return output_path
    else:
        raise FileNotFoundError(f"File not found: {input_path}")

@dsl.component(base_image='kartikjena33/kfp:1.0.5')
def hyperparameter_tuning(input_path: str) -> dict:
    import pandas as pd
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.linear_model import LogisticRegression

    data = pd.read_csv(input_path)
    X = data.drop('quality', axis=1)
    y = data['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model
    model = LogisticRegression()

    # Define the hyperparameters and their values
    param_grid = {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'saga']
    }

    # Perform grid search
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")

    return best_params

@dsl.component(base_image='kartikjena33/kfp:1.0.5')
def train_model(input_path: str, best_params: dict, model_path: dsl.Output[dsl.Artifact]):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    import joblib
    import os

    data = pd.read_csv(input_path)
    X = data.drop('quality', axis=1)
    y = data['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model with the best parameters
    model = LogisticRegression(**best_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy}')

    # Specify the filename for the model
    model_filename = os.path.join(model_path.path, 'winequality_model.pkl')

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)

    joblib.dump(model, model_filename)
    model_path.metadata['accuracy'] = accuracy

@dsl.component(base_image='kartikjena33/kfp:1.0.5')
def test_model(model_path: dsl.Input[dsl.Artifact]) -> str:
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    import joblib
    import os

    # Load the model
    model_filename = os.path.join(model_path.path, 'winequality_model.pkl')
    model = joblib.load(model_filename)

    # Prepare test data
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    data = pd.read_csv(url, sep=';')
    X_test = data.drop('quality', axis=1).sample(5)  # Sample 5 rows for testing
    y_test = data['quality'].sample(5)  # Sample the corresponding labels

    # Make predictions
    y_pred = model.predict(X_test)

    # Compare predictions with actual labels
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Predictions: {y_pred}")
    print(f"Actual labels: {y_test.values}")
    print(f"Accuracy: {accuracy}")

    return "Model testing completed."

@dsl.pipeline(
    name='Wine Quality Pipeline with Hyperparameter Tuning',
    description='A pipeline to ingest, preprocess, tune hyperparameters, train, and test a model on the wine quality dataset.'
)
def wine_quality_pipeline():
    ingest_task = ingest_data().set_caching_options(False)
    preprocess_task = preprocess_data(input_path=ingest_task.output).set_caching_options(False)
    tuning_task = hyperparameter_tuning(input_path=preprocess_task.output).set_caching_options(False)
    train_task = train_model(input_path=preprocess_task.output, best_params=tuning_task.output).set_caching_options(False)
    test_task = test_model(model_path=train_task.outputs['model_path']).set_caching_options(False)

    kubernetes.mount_pvc(
        ingest_task,
        pvc_name='my-pvc',
        mount_path='/data',
    )
    kubernetes.mount_pvc(
        preprocess_task,
        pvc_name='my-pvc',
        mount_path='/data',
    )
    kubernetes.mount_pvc(
        tuning_task,
        pvc_name='my-pvc',
        mount_path='/data',
    )
    kubernetes.mount_pvc(
        train_task,
        pvc_name='my-pvc',
        mount_path='/data',
    )
    kubernetes.mount_pvc(
        test_task,
        pvc_name='my-pvc',
        mount_path='/data',
    )

if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(wine_quality_pipeline, 'wine_quality_pipeline.yaml')