# Spam Detector Network

## Table of Contents
1. [Project Description](#project-description)
2. [Project Structure](#project-structure)
3. [Model Characteristics](#model-characteristics)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Training and Evaluation](#model-training-and-evaluation)
6. [Tools and Technologies Used](#tools-and-technologies-used)
7. [Server Setup and Deployment](#server-setup-and-deployment)
8. [Node-RED Workflow](#node-red-workflow)
9. [Conclusion](#conclusion)
10. [Future Proposals](#future-proposals)
11. [Authors](#authors)
12. [License](#license)
13. [Acknowledgment](#acknowledgment)

---

## Project Description

In 2023, approximately 45.6% of global emails were identified as spam, marking a decrease from nearly 49% in 2022. Despite this decline, spam remains a critical issue in email communications. This project focuses on developing an effective spam email detection system using deep learning techniques, with a particular emphasis on leveraging a pre-trained DistilBERT model.

### Project Objectives

- **Developing a Robust Model**: Utilize state-of-the-art deep learning models for accurate classification of spam and non-spam emails.
- **Efficiency in Processing**: Implement efficient preprocessing steps to handle large volumes of email data.
- **Real-Time Detection**: Deploy the model for real-time spam detection in email systems.
- **Integration with Node-RED**: Integrate the spam detection model with Node-RED for seamless workflow automation.

## Project Structure

### Dataset Used

- **Enron Spam Dataset**
  - **Description**: Contains emails from Enron Corporation labeled as spam or non-spam.
  - **Link**: [Enron Spam Dataset](https://github.com/MWiechmann/enron_spam_data)

### Model Building

- Utilized a pre-trained DistilBERT model, customized for binary classification by incorporating a ReLU activation layer, Dropout layer, and final linear layer.

### Training and Validation

- Hyperparameters tuned through validation set to find optimal settings.
- Combined training and validation sets for final model training to minimize overfitting and maximize performance.

### Performance Evaluation

- Evaluated model using metrics like accuracy, precision, recall, and F1-score.
- Achieved an accuracy of 97% with strong recall, indicating minimal false positives and false negatives.

### Deployment and Monitoring

- Deployed trained model in production for real-time spam detection.
- Monitored model performance and server uptime for reliability.

### File Structure

- `SpamDetectionNetwork.ipynb`: Jupyter notebook for model development and training.
- `Prediction-Server.py`: Python script for setting up server for real-time spam prediction.
- `SpamDetector-NodeRed.json`: Node-RED configuration file for integration and deployment.
- `Model/`: Directory containing saved model files.
  - `saved_model.pb`: TensorFlow model file.
  - `variables/`: Directory for model variables.

## Model Characteristics

The project selected DistilBERT for its efficient training capabilities and reduced size. Customized model architecture replaced final classification layer with ReLU activation, Dropout for regularization, and linear layer for binary classification. Frozen feature extraction retained pre-trained weights for focused fine-tuning.

## Data Preprocessing

1. **Data Reading and Integration**:
   - Reads email data from a CSV file.
   - Merges 'Subject' and 'Message' columns into a unified 'Text' column for streamlined text processing.

2. **Label Encoding**:
   - Converts categorical labels ('Spam' and 'Ham') into binary format (1 for 'spam', 0 for 'ham') to facilitate classification.

3. **Data Cleaning and Handling**:
   - Removes unnecessary columns like 'Date', 'Subject', and 'Message ID' to simplify the dataset.
   - Handles missing values to ensure data completeness.

4. **Text Preprocessing**:
   - Utilized `googletrans` for translating non-English text to English, ensuring uniform processing and analysis across different languages.
   - Cleans text by removing non-alphanumeric characters and other noise.
   - Employed DistilBERT's tokenizer to tokenize text efficiently, ensuring compatibility with model input requirements.
   - Removes stopwords and lemmatizes tokens for improved analysis.

### Data Loader

Implemented `DistilBERTDataset` class for handling input during model training and validation. Utilized DistilBERT tokenizer for tokenization, padding, and truncation, preprocessing CSV data efficiently via PyTorch's DataLoader with multiprocessing support.

## Model Training and Evaluation

### Training Process

- Utilized AdamW optimizer for training with CrossEntropyLoss as the loss function.
- Monitored training progress through epochs, saving best model weights based on validation accuracy and loss.

### Validation and Testing

- Validated model performance on a separate test dataset to ensure generalizability and accuracy.
- Evaluated metrics such as precision, recall, and F1-score to assess model effectiveness in spam detection.

### Attention Masking

During dataset preparation, each input sequence was tokenized and converted into an ID sequence. To ensure computational efficiency and handle varying sequence lengths, attention masks were applied during testing. These masks identify relevant tokens (set to 1) and irrelevant tokens (set to 0) during attention calculation.

### Model Testing and Results

Achieved an accuracy of 97% with a robust recall score, indicating effective spam detection with minimal false positives and negatives. Detailed error analysis included confusion matrices and tokenized misclassifications, improving model understanding and performance.

## Tools and Technologies Used

- **Deep Learning Framework**: PyTorch for developing and training deep learning models.
- **Model Architecture**: DistilBERT for efficient text classification tasks.
- **Data Handling**: Pandas and NumPy for data manipulation and preprocessing.
- **Visualization**: Matplotlib for visualizing training progress and model evaluation metrics.
- **Collaborative Development**: Google Colab for cloud-based Jupyter environment and GPU utilization.
- **Deployment**: Node-RED for workflow automation and integration, Flask for creating API endpoints.

## Server Setup and Deployment

### Setting Up Prediction Server

- Implemented Flask application for hosting the trained model, enabling real-time predictions via HTTP requests.
- Utilized Docker for containerization, ensuring portability and scalability of the prediction server.

### Monitoring and Maintenance

- Integrated monitoring tools to track model performance and server uptime, ensuring continuous operation and reliability.
- Implemented logging mechanisms to capture errors and user interactions for troubleshooting and improvement.

## Node-RED Workflow

### Workflow Integration

- Developed Node-RED flows to automate email processing and spam detection.
- Integrated with IMAP servers for retrieving emails, processing through Flask API for prediction, and routing based on spam classification.

### Real-Time Decision Making

- Configured decision-making nodes to classify emails as spam or non-spam based on model predictions.
- Automated responses for spam emails, ensuring efficient email management and user communication.

## Conclusion

The Spam Detector Network project encompasses a comprehensive approach to building and deploying a spam detection system using advanced machine learning techniques. The integration with Node-RED enhances automation capabilities, making it suitable for real-time email spam detection and management.

## Future Proposals

- **Cloud Deployment**: Deploying the model on a cloud server using Docker containers for improved scalability and management.
- **Big Data Handling**: Implementing Hadoop or Spark frameworks for handling large volumes of data and advanced result analysis.
- **Mobile Application**: Developing a cross-platform mobile app using Flutter for local model execution or Swift for optimized iOS performance.

## Authors

- [Leonardo Catello](https://github.com/Leonard2310)
- [Lorenzo Manco](https://github.com/Rasbon99)
- [Aurora D'Ambrosio](https://github.com/AuroraD-99)

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE). Refer to the LICENSE file for more information.

## Acknowledgment

Gratitude to [Marcel Wiechmann](https://github.com/MWiechmann), creator of the Enron Spam Dataset, for providing valuable data for this project.
