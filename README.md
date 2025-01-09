
- `cse439_hw01_201805069.py`: This script contains all the code for loading, preprocessing, training, and evaluating the models.
- `model_results_with_validation.csv`: This file contains the results of the trained models, including various metrics and hyperparameters used.

## How to Run

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd Fruit-Image-Classification
    ```

2.  **Install required libraries:**

    ```bash
    pip install numpy pandas seaborn matplotlib scikit-learn xgboost lightgbm tensorflow opencv-python
    ```
    or you can use:
     ```bash
    pip install -r requirements.txt
    ```
3.  **Set your data paths:**

    Modify the `data_path` variable in `cse439_hw01_201805069.py` to point to your local directory where the Fruits-360 dataset is located.

    ```python
    data_path = r"C:\\Users\\tamer\\Desktop\\Deep Learning\\fruits-360_dataset_original-size\\fruits-360-original-size"
    ```
4.  **Run the script:**

    ```bash
    python cse439_hw01_201805069.py
    ```

## Key Findings

- The project shows the performance of various machine learning and deep learning models on the Fruits 360 dataset.
- Different hyperparameter settings such as photo size, PCA component number, and dropout rate have an impact on the results
- The effect of dropout on the performance of ANN and CNN models are evaluated.
- The confusion matrix for the Random Forest model is visualized.
- The accuracy and loss graphs of ANN and CNN models are visualized.

## Results

The results of the experiments are saved in `model_results_with_validation.csv`. This file contains metrics such as accuracy, precision, recall, F1 score, cross-validation scores, and training times for each model and parameter setting. You can also find the generated plots in the output of the python code.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing

Feel free to fork this repository and contribute by submitting a pull request.

## Contact

For any questions or feedback, feel free to contact:
Tamer Kanak
tamerkanak75@gmail.com
