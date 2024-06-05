# Spam Detector Network

## Table of Contents
1. [Project Description](#project-description)
2. [Project Structure](#project-structure)
3. [Model Characteristics](#model-characteristics)
4. [Tools and Technologies Used](#tools-and-technologies-used)
5. [Conclusion](#conclusion)
6. [Authors](#authors)

## Project Description

In 2023, nearly 45.6% of all emails worldwide were identified as spam, down from almost 49% in 2022. Despite the decrease, spam remains a significant concern in email communication. The project focuses on building a robust spam email detection system using deep learning techniques, with a particular emphasis on leveraging a pre-trained BERT model.

### Preprocessing Steps:
- **Tokenization**: Breaking down emails into individual words or tokens.
- **Stemming**: Reducing words to their root forms to normalize text.
- **Other Text Transformations**: Additional preprocessing steps were applied to enhance the quality of input data.

## Project Structure

### Dataset Used

- **Enron Spam Dataset**
  - **Description**: Contains emails from the Enron Corporation, labeled as spam or non-spam.
  - **Link**: [Enron Spam Dataset](https://github.com/MWiechmann/enron_spam_data)

### Technical Implementation

#### Data Collection

- A comprehensive dataset of labeled emails (spam and non-spam) was gathered from the Enron Spam Dataset.

#### Data Preprocessing

- Emails underwent preprocessing steps including tokenization, stemming, and other text transformations to prepare them for model training.

#### Model Building

- Utilized a pre-trained BERT (Bidirectional Encoder Representations from Transformers) model.
- Fine-tuned the pre-trained BERT model by training the last few fully connected layers.

#### Training and Validation

- The model was trained on a training set and validated on a separate test set to assess its performance.

#### Performance Evaluation

- Evaluated the model's performance using standard metrics such as accuracy, precision, recall, and F1-score.
- Analyzed false positives and false negatives to identify areas for improvement.

#### Deployment and Monitoring

- Deployed the trained model in a production environment for real-time spam detection.
- Implemented monitoring mechanisms to track model performance and gather feedback for continuous improvement.

## Model Characteristics

- **High Accuracy and Precision**: The model aims to accurately classify emails as spam or non-spam with minimal errors.
- **Robustness**: The model should be able to handle various types of spam emails and adapt to changes in spamming techniques.
- **Efficiency**: Real-time processing capabilities are crucial to ensure timely spam detection without significant delays.
- **Generalization**: The model should generalize well to unseen data, ensuring consistent performance in different email environments.

## Tools and Technologies Used

- **PyTorch**: Utilized for deep learning model development and training.
- **BERT (Bidirectional Encoder Representations from Transformers)**: Leveraged a pre-trained BERT model for natural language understanding tasks.
- **Google Colab**: Used Colab, a cloud-based Jupyter notebook environment, for model development and training.

## Conclusion

Building an effective spam email detection system requires a combination of advanced deep learning techniques, thorough data preprocessing, and rigorous model evaluation. By leveraging pre-trained models like BERT and fine-tuning them for specific tasks, we can develop robust solutions capable of combating spam effectively.

## Authors

- Leonardo Catello
- Lorenzo Manco
- Aurora D’Ambrosio

## Acknowledgment
We would like to express our gratitude to [Marcel Wiechmann](#https://github.com/MWiechmann) the creator of the Enron Spam Dataset for providing valuable data for our project.