📦 VisionClassify is a deep-learning project built with TensorFlow/Keras to classify animal images using a custom Convolutional Neural Network (CNN). The project includes dataset preprocessing, image augmentation, model training, evaluation, and prediction scripts. Everything is modular, reproducible, and ready for deployment or experimentation.

🚀 Features

Clean modular code (data loading, model building, training, predicting)

Image augmentation for stronger learning

CNN model with >75 epochs training pipeline

Accuracy & loss visualization

Prediction script for custom images

Notebook versions for exploration and training

🗂 Project Structure
VisionClassify/
│── data/
│── notebooks/
│── src/
│── outputs/
│── README.md
│── requirements.txt
│── .gitignore

📊 Model Summary

3 convolution blocks (32, 64, 128 filters)

MaxPooling layers for downsampling

Dense classifier with dropout

Softmax output for multi-class classification

🧪 Notebooks Included

exploration.ipynb – Dataset analysis & visualization

model_training.ipynb – Full training workflow

🔧 Install Dependencies
pip install -r requirements.txt

▶ Run Training
python src/train.py

🔍 Make Predictions
python src/predict.py --img path/to/image.jpg

📬 Contribution

PRs and suggestions are welcome.

🛡 License

MIT License.