#%%
import mlflow
import pandas as pd

model_name = "production"
model_version = "latest"

# Load the model from the Model Registry
model_uri = f"models:/{model_name}/{model_version}"
logged_model = mlflow.sklearn.load_model(model_uri)

# GENERATE PREDICTION DATA ---------------------

def create_data(
    sex: str = "female",
    age: float = 29.0,
    fare: float = 16.5,
    embarked: str = "S",
) -> str:
    """
    """

    df = pd.DataFrame(
        {
            "Sex": [sex],
            "Age": [age],
            "Fare": [fare],
            "Embarked": [embarked],
        }
    )

    return df


data = pd.concat(
    [create_data(age=40), create_data(sex="male")]
)

# PREDICTION ---------------------

logged_model.predict(pd.DataFrame(data))

# %%
