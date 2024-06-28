# Spam Detector Network

## Table of Contents
1. [Project Description](#project-description)
2. [Project Structure](#project-structure)
3. [Model Characteristics](#model-characteristics)
4. [Tools and Technologies Used](#tools-and-technologies-used)
5. [Server Setup and Deployment](#server-setup-and-deployment)
6. [Node-RED Workflow](#node-red-workflow)
7. [Conclusion](#conclusion)
8. [Authors](#authors)
9. [License](#license)
10. [Acknowledgment](#acknowledgment)

## Project Description

In 2023, nearly 45.6% of all emails worldwide were identified as spam, down from almost 49% in 2022. Despite the decrease, spam remains a significant concern in email communication. This project focuses on building a robust spam email detection system using deep learning techniques, with a particular emphasis on leveraging a pre-trained DistilBERT model.

### Preprocessing Steps:
1. **Data Reading and Integration**: 
   - Reads a CSV file containing email data.
   - Merges the 'Subject' and 'Message' columns into a single 'Text' column for unified text processing.

2. **Label Encoding**: 
   - Converts categorical labels ('Spam' and 'Ham') into binary format (1 for 'spam', 0 for 'ham') to facilitate classification tasks.

3. **Data Cleaning and Handling**:
   - Removes unnecessary columns like 'Date', 'Subject', 'Message ID' to streamline the dataset.
   - Handles missing values in the dataset to ensure completeness.

4. **Text Preprocessing**:
   - Translates non-English text to English using automated translation.
   - Cleans the text by removing non-alphanumeric characters and other noise.
   - Tokenizes the text using the DistilBERT tokenizer for natural language processing tasks.
   - Removes stopwords and lemmatizes tokens to reduce them to their base form for better analysis.

These preprocessing steps are essential to enhance the quality and consistency of the input data before feeding it into machine learning models. They ensure that the text data is standardized, cleaned, and optimized for subsequent analysis and model training processes. This systematic approach prepares the dataset effectively for tasks such as classification of emails into spam and non-spam categories.

## Project Structure

### Dataset Used

- **Enron Spam Dataset**
  - **Description**: Contains emails from the Enron Corporation, labeled as spam or non-spam.
  - **Link**: [Enron Spam Dataset](https://github.com/MWiechmann/enron_spam_data)


#### Model Building

- Utilized a pre-trained DistilBERT (a distilled version of BERT) model.
- Modified the DistilBERT model for binary classification by adding a ReLU activation layer, a Dropout layer, and a final linear layer.

#### Training and Validation

- The model's feature extraction layers were frozen, and only the classification layer was trained.
- The model was trained on a training set and validated on a separate test set to assess its performance.

#### Performance Evaluation

- Evaluated the model's performance using standard metrics such as loss and accuracy.
- Visualized training progress and results using appropriate plots.

#### Deployment and Monitoring

- Deployed the trained model in a production environment for real-time spam detection.
- Implemented monitoring mechanisms to track model performance and gather feedback for continuous improvement.

### File Structure

- `SpamDetectionNetwork.ipynb`: Jupyter notebook containing the model development and training code.
- `Prediction-Server.py`: Python script to set up a server for real-time spam prediction.
- `SpamDetector-NodeRed.json`: Configuration file for Node-RED to facilitate integration and deployment.
- `Model/`: Directory containing the saved model files.
  - `saved_model.pb`: The saved TensorFlow model.
  - `variables/`: Directory containing model variables.
- `SpamDetection-Team01.pdf`: Project report detailing the methodologies and findings.

## Model Characteristics

For this project, the DistilBERT model for sequence classification was chosen due to its reduced size and efficient training capabilities. The model's architecture was customized by replacing the final classification layer with a sequence comprising a ReLU activation layer, a Dropout layer for regularization, and a linear layer for binary classification. The feature extraction layers of the DistilBERT model were frozen to retain pre-trained weights and focus on fine-tuning for the specific classification task.

## Data Loader

To handle data input during model training and validation, a custom `DistilBERTDataset` class was implemented. This class utilizes the DistilBERT tokenizer for text tokenization, padding, and truncation to fit the required maximum sequence length. It preprocesses text data from CSV files containing training and validation datasets, enabling efficient batch loading using PyTorch's DataLoader with support for multiprocessing.

## Training with Validation

The training and validation process involves organizing epochs where the model processes batches of training data, computes outputs, calculates loss against true labels, and updates parameters via backpropagation using the AdamW optimizer. Validation occurs after each epoch to evaluate model performance on a separate dataset, saving the best model weights based on validation accuracy and loss. The process uses CrossEntropyLoss as the loss function and supports learning rate scheduling for optimization.

## Tools and Technologies Used

- PyTorch: Used for developing and training deep learning models, specifically for natural language understanding tasks in this context.
- DistilBERT: Utilized a pre-trained DistilBERT model, which is a distilled version of BERT (Bidirectional Encoder Representations from Transformers), for efficient NLP tasks such as text classification.
- Pandas and NumPy: Employed for data manipulation, preprocessing, and handling structured data from CSV files.
- Matplotlib: Used to visualize training progress, model evaluation metrics, and other insights during the development phase.
- Google Colab: Utilized as a cloud-based Jupyter notebook environment for collaborative model development, training on GPU resources, and experimentation.
- Node-RED: Integrated as a flow-based development tool for visual programming, facilitating data flow and integration tasks. Specifically, Node-RED is used to handle HTTP POST requests for model predictions.
- Flask: Integrated Flask to create a local server where Node-RED sends HTTP POST requests to obtain predictions from the trained model. Flask enables easy deployment and hosting of the machine learning model as an API endpoint.
- TensorFlow: Used for saving and exporting the trained model, ensuring compatibility and ease of deployment in various environments.

## Server Setup and Deployment

### Installing Necessary Packages

To set up the server, it is essential to ensure that the required packages are installed. This includes uninstalling any existing TensorFlow installations and installing the specific version used in this project, along with other dependencies such as `langdetect`, `googletrans`, and `transformers`.

### Initializing the Flask App and Loading the Model

The server is initialized using Flask, a lightweight WSGI web application framework in Python. The trained TensorFlow model is loaded using specific options to ensure compatibility and performance.

### Text Translation and Preprocessing

To handle multi-language emails, the project includes functionality to detect and translate text to English using `langdetect` and `googletrans`. Additionally, text preprocessing steps such as tokenization, cleaning, and lemmatization are performed to prepare the email content for prediction.

### Defining the Prediction Route

A route for making predictions is defined in the Flask app. This route processes incoming requests, preprocesses the email text, performs the prediction using the loaded TensorFlow model, and returns the prediction result in JSON format.

## Node-RED Workflow

### Workflow Overview

The Node-RED workflow facilitates the integration of the spam detection system with email services. The workflow includes components for receiving emails, sending predictions to the Flask server, processing the predictions, and taking appropriate actions based on the prediction results.

### Key Components

1. **Email Receiver**: Retrieves emails from an IMAP server at regular intervals.
2. **HTTP Request**: Sends the email content to the Flask server for spam prediction.
3. **Process Prediction**: Processes the response from the Flask server to determine the appropriate action.
4. **Switch on Prediction**: Directs the workflow based on the prediction result.
5. **Spam Warning and Email Sender**: Sends a warning or processes the email further based on the prediction.

### Example Workflow

The workflow can be triggered manually or automatically to handle incoming emails, make predictions, and take actions such as sending spam warnings or forwarding non-spam emails.

## Conclusion

The Spam Detector Network project demonstrates a comprehensive approach to building and deploying a spam detection system using advanced machine learning techniques and modern tools like Flask and Node-RED. The system is designed to be robust, efficient, and capable of handling real-time email spam detection.

## Authors
- [Leonardo Catello](https://github.com/Leonard2310)
- [Lorenzo Manco](https://github.com/Rasbon99)
- [Aurora D'Ambrosio](https://github.com/AuroraD-99)

## License
This project is licensed under the [GNU General Public License v3.0](LICENSE). Refer to the LICENSE file for more information.

## Acknowledgment
We would like to express our gratitude to [Marcel Wiechmann](https://github.com/MWiechmann), the creator of the Enron Spam Dataset, for providing valuable data for our project.
