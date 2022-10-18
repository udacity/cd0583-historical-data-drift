# Historical Data Drift

In this tutorial, we will calculate and visualize historical data drift, which tells us how data has changed. We have used the [UCI Bike Sharing dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset) for this tutorial.

### Steps to run
1. Clone this repository.
2. Install dependencies from the `requirements.txt` file.
3. The `expo.py` file contains the code to calculate data drift (the process is the same as the previous tutorial). You can run the file and setup an mlflow server as follows:
    ```bash
    python expo.py & mlflow ui
    ```

### What is expo.py?
It contains the code to calculate data and feature drift using **evidently**, generate visualizations using **plotly**, and log the results using **mlflow**.
