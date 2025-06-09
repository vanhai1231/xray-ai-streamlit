# 🏪 X-ray AI Streamlit App

A web application using **Streamlit** to classify **chest X-ray images** as *PNEUMONIA* or *NORMAL*
based on a pretrained model hosted on Hugging Face:
👉 [vanhai123/simple-cnn-chest-xray](https://huggingface.co/vanhai123/simple-cnn-chest-xray)

---

## 🚀 Features

* Upload a chest X-ray image (JPG/PNG)
* Predicts pneumonia status using a CNN model
* Displays the predicted label with confidence score
* Simple UI using Streamlit

---

## 🧪 Model Information

* Architecture: Simple CNN (3 conv layers)
* Framework: PyTorch
* Dataset: Chest X-ray images (COVIDx / Pneumonia dataset)
* Hosted on Hugging Face: [https://huggingface.co/vanhai123/simple-cnn-chest-xray](https://huggingface.co/vanhai123/simple-cnn-chest-xray)

---

## 📦 Installation

```bash
git clone https://github.com/vanhai1231/xray-ai-streamlit.git
cd xray-ai-streamlit
pip install -r requirements.txt
streamlit run app.py
```

---

## 🌐 Deploy Online

You can deploy this app using:

* [Streamlit Cloud](https://share.streamlit.io)
* [Hugging Face Spaces](https://huggingface.co/spaces)

---

## 🖼️ Screenshot

![demo](https://user-images.githubusercontent.com/your_placeholder_image.png)

---

## 👨‍⚕️ Example Prediction

| Input Image | Output Label | Confidence |
| ----------- | ------------ | ---------- |
| X-ray 1     | PNEUMONIA    | 94.7%      |
| X-ray 2     | NORMAL       | 98.2%      |

---

## 📜 License

MIT License

---

*Developed by [vanhai123](https://huggingface.co/vanhai123) | Model powered by Hugging Face Hub*
