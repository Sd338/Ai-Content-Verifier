![Screenshot 2024-09-28 165852](https://github.com/user-attachments/assets/cc59d92e-d123-4221-8df4-68fcd5fa09ae)


# 📄 AI Content Verifier

## 📜 Project Overview
**AI Content Verifier** is a strong tool that helps tell apart text written by humans and text created by AI. As AI gets better at writing like humans, it’s important to check if the text is real. This tool is helpful for things like checking content, academic honesty, and keeping information accurate.

The project uses smart machine learning methods and natural language processing to look at written data, giving users clear results and confidence scores. The easy-to-use graphical interface makes it simple for researchers, teachers, and content creators. With adjustable settings and detailed logs, users can improve detection and keep track of their results easily.

## ⭐ Key Features
- **AI Detection**: Accurately classifies text as AI-generated or human-written, helping users verify content authenticity.
- **Confidence Scoring**: Provides confidence scores for each classification to assess the reliability of results.
- **User-Friendly GUI**: An intuitive graphical user interface simplifies content analysis, making it easy for users without technical expertise.
- **Customizable Settings**: Allows users to adjust parameters and configurations to enhance detection performance based on their specific needs.
- **Robust Logging**: Comprehensive logs of predictions, training processes, and user interactions to facilitate better analysis.
- **Cross-Platform Compatibility**: Runs on Windows, macOS, and Linux, ensuring accessibility to a wide range of users.
- **Efficient Data Handling**: Utilizes CSV files for dataset management, simplifying data import, export, and manipulation.
- **Scalable Model**: Designed to accommodate additional training data and improvements, ensuring the verifier evolves with AI advancements.

## 🛠️ Technologies Used
- **Programming Language**: Python
- **Machine Learning Libraries**:
  - scikit-learn: For implementing machine learning algorithms and preprocessing.
  - pandas: For data manipulation and analysis.
  - numpy: For numerical computations.
- **Natural Language Processing**:
  - NLTK: For text processing and linguistic analysis.
  - SpaCy: For advanced NLP tasks and efficient text processing.
- **GUI Framework**:
  - Tkinter: For creating the graphical user interface.
  - PyQt (optional): An alternative GUI framework.
- **Data Handling**: CSV format for dataset management and storage.
- **Version Control**: Git for source code management.

## 📂 Project Structure
Here's the structure of the project directory:

```plaintext
AI-Content-Verifier/
├── .vscode/                          # VS Code configuration files
├── config/                           # Configuration files
│   └── config.yaml                   # Configuration settings
├── data/                             # Data directory
│   ├── processed/                    # Processed datasets for training
│   └── raw/                          # Raw datasets
├── logs/                             # Log files
│   └── training.log                  # Log of model training processes
├── src/                              # Source code directory
│   ├── data/                         # Data processing scripts
│   │   └── preprocess_data.py        # Script for data preprocessing
│   ├── modeling/                     # Model training and evaluation scripts
│   │   ├── evaluate_model.py         # Script to evaluate the model
│   │   ├── predict.py                # Script for making predictions
│   │   └── train_model.py            # Script for training the model
│   ├── saved_models/                 # Directory for saved models
│   │   ├── best_model.pkl            # Best trained model for predictions
│   │   ├── model_logistic_regression.pkl # Logistic regression model
│   │   └── tfidf_vectorizer.pkl      # TF-IDF Vectorizer for feature extraction
│   └── utils/                        # Utility scripts
│       └── helpers.py                # Utility functions and helpers
├── .gitignore                        # Git ignore file
├── content_verifier_model.py         # Main logic for content verification
├── gui.py                            # GUI interface for the project
├── LICENSE                           # License information
├── predictions_log.txt               # Log of predictions made by the model
├── README.md                         # Project documentation (this file)
├── requirements.txt                  # Required Python packages for the project
└── test_helpers.py                   # Unit tests for helper functions
```

## 📦 Installation Guide
1. **Clone the Repository**:
   Open your terminal or command prompt and run the following command to clone the repository:
   ```bash
   git clone https://github.com/your_username/AI-Content-Verifier.git
   cd AI-Content-Verifier
   ```

2. **Install Required Packages**:
   Use `pip` to install the necessary dependencies from the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the Application**:
   Modify the `config/config.yaml` file to set your preferences for model parameters and other settings.

4. **Set Up Data Directories**:
   Ensure the raw and processed datasets are placed in the following directories:
   - Raw datasets: `data/raw/`
   - Processed datasets: `data/processed/`

5. **Run the Application**:
   To run the GUI application, use the following command:
   ```bash
   python gui.py
   ```

## 📊 Data Model Training
The **AI Content Verifier** uses raw and processed datasets for training. Follow these steps to prepare and train your model:

1. **Prepare Datasets**:
   - Place the raw and processed datasets in the appropriate directories:
     - Raw datasets: `data/raw/`
     - Processed datasets: `data/processed/`

2. **Train the Model**:
   To start training, run the following command:
   ```bash
   python src/modeling/train_model.py
   ```

3. **Monitor Training**:
   Logs will be generated in the `logs/` directory to help monitor the training process and performance metrics.

4. **Save the Trained Model**:
   After training, the model will be saved in the `src/saved_models/` directory for future predictions.

Ensure the configurations in `config/config.yaml` are set properly to optimize training performance.

## 🤝 Contributing
We welcome contributions to improve the **AI Content Verifier**! Here's how you can contribute:

1. **Fork the Repository**:
   - Click the "Fork" button at the top right of the project page on GitHub.

2. **Clone Your Fork**:
   ```bash
   git clone https://github.com/your-username/AI-Content-Verifier.git
   cd AI-Content-Verifier
   ```

3. **Create a New Branch**:
   - Create a branch for your feature or bug fix:
   ```bash
   git checkout -b feature-or-bugfix-name
   ```

4. **Make Your Changes**:
   - Implement your changes.

5. **Commit Your Changes**:
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

6. **Push to Your Branch**:
   ```bash
   git push origin feature-or-bugfix-name
   ```

7. **Submit a Pull Request**:
   - Go to the original repository on GitHub and submit a pull request with a clear description.

**Guidelines**:
- Follow the existing code style.
- Write clear commit messages.
- Test your changes before submitting a pull request.

Thank you for contributing!

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## 📧 Contact
For questions or support, please reach out via the contact methods on my GitHub profile. The email address in the GUI (`support@contentverifier.com`) is not real and is used for demonstration purposes only, so please don't send emails to that address.


