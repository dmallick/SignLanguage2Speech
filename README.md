# SignLanguage2Speech
MVP: Recognize static hand signs convert them to text, and speak them out using a TTS engine.


Notes:
I am using python 3.9.6 as part of the venv
✅ Step 2: Train a Model on Sign Language MNIST
We'll use a publicly available dataset of static ASL signs (A–Y except J and Z, since they involve motion) to train a CNN classifier.
📥 Step 3: Download Dataset
    1. sign_mnist_train.csv
    2. sign_mnist_test.csv
🤖 Step 4: Train a CNN Model
✅ Step 5: Real-Time Sign Prediction + TTS
    1. Capture hand using webcam
    2. Convert hand region to 28x28 grayscale image
    3. Use model to predict → convert to letter → speak it 
How It Works
Feature	Description
📷 Webcam	Captures hand gestures
🧠 Model	Predicts letter (A–Y)
🪢 Buffer	Stores stable letter sequence
⏱️ Auto Speak	When hand disappears for 2 seconds, the word is spoken
🔊 TTS Output	Uses pyttsx3 to speak the full word


