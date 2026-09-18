import mlflow
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss


# Set MLflow experiment
mlflow.set_experiment("Student_Model_Experiment")

# Load Wine classification dataset
wine = load_wine()
X = wine.data
y = wine.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Two different model configurations
experiments = [
    {
        "name": "Experiment_1",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 200
    },
    {
        "name": "Experiment_2",
        "learning_rate": 0.01,
        "batch_size": 16,
        "epochs": 200
    }
]

for config in experiments:

    with mlflow.start_run(run_name=config["name"]):

        # Create model using the experiment configuration
        model = MLPClassifier(
            learning_rate_init=config["learning_rate"],
            batch_size=config["batch_size"],
            max_iter=config["epochs"],
            random_state=42
        )

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        loss = log_loss(y_test, probabilities)

        # Log parameters
        mlflow.log_param("learning_rate", config["learning_rate"])
        mlflow.log_param("batch_size", config["batch_size"])
        mlflow.log_param("epochs", config["epochs"])

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("loss", loss)

        # Create model information file
        with open("model_info.txt", "w") as file:
            file.write(f"Run Name: {config['name']}\n")
            file.write(f"Learning Rate: {config['learning_rate']}\n")
            file.write(f"Batch Size: {config['batch_size']}\n")
            file.write(f"Epochs: {config['epochs']}\n")
            file.write(f"Accuracy: {accuracy:.4f}\n")
            file.write(f"Loss: {loss:.4f}\n")

        # Log model_info.txt as an MLflow artifact
        mlflow.log_artifact("model_info.txt")

        # Log the trained model

        print(f"{config['name']} completed")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Loss: {loss:.4f}")
        print("-" * 40)
