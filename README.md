Glaucoma Disease Detection

This project is a **Glaucoma Detection System** that analyzes eye images and predicts whether glaucoma is present using machine learning techniques.



 Features

* Detects glaucoma from retinal images
* User-friendly interface
* Fast and accurate predictions
* Built using Python and ML models

 Tech Stack

* Python
* OpenCV
* NumPy
* TensorFlow / Keras *(if used)*
* Streamlit *(if UI is used)*

---

 Project Structure

```
Glaucoma-disease-detection/
│── dataset/
│── model/
│── app.py
│── requirements.txt
│── README.md
```

---

How to Run the Project in VS Code

Step 1: Clone the Repository

```bash
git clone <your-repo-link>
cd Glaucoma-disease-detection
```

---

Step 2: Open in VS Code

* Open **VS Code**
* Click on **File → Open Folder**
* Select the project folder

---

Step 3: Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it:

* Windows:

```bash
venv\Scripts\activate
```

---

 Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

---

Step 5: Run the Project

If using Streamlit:

```bash
streamlit run app.py
```

OR normal Python:

```bash
python app.py
```

---

📊 Output

* Upload an eye image
* Model predicts: **Glaucoma / No Glaucoma**

---

Notes

* Make sure Python is installed
* Install required libraries before running
* Keep dataset and model files in correct folders

---

 Future Improvements

* Improve model accuracy
* Add real-time camera detection
* Deploy as a web app

---

 License

This project is for educational purposes only.
