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

---

## Project Description

In 2023, approximately 45.6% of global emails were identified as spam, marking a decrease from nearly 49% in 2022. Despite this decline, spam remains a critical issue in email communications. This project focuses on developing an effective spam email detection system using deep learning techniques, with a particular emphasis on leveraging a pre-trained DistilBERT model.

### Preprocessing Steps:

1. **Data Reading and Integration**:
   - Reads email data from a CSV file.
   - Merges 'Subject' and 'Message' columns into a unified 'Text' column for streamlined text processing.

2. **Label Encoding**:
   - Converts categorical labels ('Spam' and 'Ham') into binary format (1 for 'spam', 0 for 'ham') to facilitate classification.

3. **Data Cleaning and Handling**:
   - Removes unnecessary columns like 'Date', 'Subject', and 'Message ID' to simplify the dataset.
   - Handles missing values to ensure data completeness.

4. **Text Preprocessing**:
   - Translates non-English text to English using automated translation.
   - Cleans text by removing non-alphanumeric characters and other noise.
   - Tokenizes text using DistilBERT tokenizer for natural language processing.
   - Removes stopwords and lemmatizes tokens for improved analysis.

These preprocessing steps are crucial to standardize, clean, and optimize input data for subsequent machine learning model training and analysis, specifically for classifying emails as spam or non-spam.

## Project Structure

### Dataset Used

- **Enron Spam Dataset**
  - **Description**: Contains emails from Enron Corporation labeled as spam or non-spam.
  - **Link**: [Enron Spam Dataset](https://github.com/MWiechmann/enron_spam_data)

### Model Building

- Utilized a pre-trained DistilBERT model, customized for binary classification by incorporating a ReLU activation layer, Dropout layer, and final linear layer.

### Training and Validation

- Frozen feature extraction layers of DistilBERT model.
- Trained classification layer on training set, validated on separate test set for performance assessment.

### Performance Evaluation

- Evaluated model using metrics like loss and accuracy.
- Utilized plots for visualizing training progress and results.

### Deployment and Monitoring

- Deployed trained model in production for real-time spam detection.
- Implemented monitoring to track model performance and gather feedback.

### File Structure

- `SpamDetectionNetwork.ipynb`: Jupyter notebook for model development and training.
- `Prediction-Server.py`: Python script for setting up server for real-time spam prediction.
- `SpamDetector-NodeRed.json`: Node-RED configuration file for integration and deployment.
- `Model/`: Directory containing saved model files.
  - `saved_model.pb`: TensorFlow model file.
  - `variables/`: Directory for model variables.
- `SpamDetection-Team01.pdf`: Project report detailing methodologies and findings.

## Model Characteristics

The project selected DistilBERT for its efficient training capabilities and reduced size. Customized model architecture replaced final classification layer with ReLU activation, Dropout for regularization, and linear layer for binary classification. Frozen feature extraction retained pre-trained weights for focused fine-tuning.

## Data Loader

Implemented `DistilBERTDataset` class for handling input during model training and validation. Utilized DistilBERT tokenizer for tokenization, padding, and truncation, preprocessing CSV data efficiently via PyTorch's DataLoader with multiprocessing support.

## Training with Validation

Training entailed epochs processing batches, computing outputs, calculating loss against true labels, and optimizing via backpropagation using AdamW optimizer. Validation post each epoch assessed model on separate dataset, saving best weights by validation accuracy and loss, employing CrossEntropyLoss and learning rate scheduling.

## Tools and Technologies Used

- PyTorch: For developing and training deep learning models, focusing on natural language understanding tasks.
- DistilBERT: Leveraged pre-trained model for efficient text classification.
- Pandas and NumPy: For data manipulation, preprocessing structured data from CSVs.
- Matplotlib: For visualizing training progress, model evaluation metrics.
- Google Colab: Cloud-based Jupyter environment for collaborative model development, GPU resource utilization.
- Node-RED: Flow-based tool for visual programming, data flow, and integration tasks, handling HTTP POST for model predictions.
- Flask: Integrated for local server, facilitating Node-RED HTTP POST to predict model. Flask eases API endpoint ML model hosting, deployment.
- TensorFlow: For saving, exporting trained model, ensuring deployment compatibility.

## Server Setup and Deployment

### Installing Necessary Packages

Ensure required package installation: uninstall existing TensorFlow, install project-specific TensorFlow version, 'langdetect', 'googletrans', 'transformers' dependencies.

### Initializing Flask App and Loading Model

Flask use for lightweight WSGI Python web app framework, model load via specific options, optimizing compatibility, performance.

### Text Translation, Preprocessing

Multi-language email handling: detect, translate to English via 'langdetect', 'googletrans'. Text preprocessing incl. tokenization, cleaning, lemmatization ready email content prediction.

### Defining Prediction Route

Flask app route setup for prediction request: process incoming, preprocess email text, predict via loaded TensorFlow model, JSON format return prediction result.

## Node-RED Workflow

### Workflow Overview

Node-RED facilitates email service, spam detection integration workflow. Components retrieve emails, send predictions Flask, process, action based prediction.

### Key Components

1. **Email Receiver**: Regularly retrieves IMAP server emails.
2. **HTTP Request**: Send email content Flask server spam prediction.
3. **Process Prediction**: Response process Flask server determine action.
4. **Switch on Prediction**: Workflow directs based prediction.
5. **Spam Warning, Email Sender**: Warning send, process email prediction.

### Example Workflow

Triggered manually or automatically, handle incoming emails, predict, action spam warning, non-spam email forward.

## Conclusion

Spam Detector Network project encompasses comprehensive approach build, deploy spam detection system, advanced machine learning, Flask, Node-RED. System robust, efficient, real-time email spam detection capable.

## Authors

- [Leonardo Catello](https://github.com/Leonard2310)
- [Lorenzo Manco](https://github.com/Rasbon99)
- [Aurora D'Ambrosio](https://github.com/AuroraD-99)

## License

This project licensed under [GNU General Public License v3.0](LICENSE). Reference LICENSE file for more information.

## Acknowledgment

Gratitude to [Marcel Wiechmann](https://github.com/MWiechmann), Enron Spam Dataset creator, providing valuable project data.
