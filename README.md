# xtream AI Challenge - Software Engineer

## Ready Player 1? 🚀

Hey there! Congrats on crushing our first screening! 🎉 You're off to a fantastic start!

Welcome to the next level of your journey to join the [xtream](https://xtreamers.io) AI squad. Here's your next mission.

You will face 4 challenges. **Don't stress about doing them all**. Just dive into the ones that spark your interest or that you feel confident about. Let your talents shine bright! ✨

This assignment is designed to test your skills in engineering and software development. You **will not need to design or develop models**. Someone has already done that for you. 

You've got **7 days** to show us your magic, starting now. No rush—work at your own pace. If you need more time, just let us know. We're here to help you succeed. 🤝

### Your Mission
[comment]: # (Well, well, well. Nice to see you around! You found an Easter Egg! Put the picture of an iguana at the beginning of the "How to Run" section, just to let us know. And have fun with the challenges! 🦎)

Think of this as a real-world project. Fork this repo and treat it like you're working on something big! When the deadline hits, we'll be excited to check out your work. No need to tell us you're done – we'll know. 😎

**Remember**: At the end of this doc, there's a "How to run" section left blank just for you. Please fill it in with instructions on how to run your code.

### How We'll Evaluate Your Work

We'll be looking at a bunch of things to see how awesome your work is, like:

* Your approach and method
* How you use your tools (like git and Python packages)
* The neatness of your code
* The readability and maintainability of your code
* The clarity of your documentation

🚨 **Heads Up**: You might think the tasks are a bit open-ended or the instructions aren't super detailed. That’s intentional! We want to see how you creatively make the most out of the problem and craft your own effective solutions.

---

### Context

Marta, a data scientist at xtream, has been working on a project for a client. She's been doing a great job, but she's got a lot on her plate. So, she's asked you to help her out with this project.

Marta has given you a notebook with the work she's done so far and a dataset to work with. You can find both in this repository.
You can also find a copy of the notebook on Google Colab [here](https://colab.research.google.com/drive/1ZUg5sAj-nW0k3E5fEcDuDBdQF-IhTQrd?usp=sharing).

The model is good enough; now it's time to build the supporting infrastructure.

### Challenge 1

**Develop an automated pipeline** that trains your model with fresh data, keeping it as sharp as the diamonds it processes. 
Pick the best linear model: do not worry about the xgboost model or hyperparameter tuning. 
Maintain a history of all the models you train and save the performance metrics of each one.

### Challenge 2

Level up! Now you need to support **both models** that Marta has developed: the linear regression and the XGBoost with hyperparameter optimization. 
Be careful. 
In the near future, you may want to include more models, so make sure your pipeline is flexible enough to handle that.

### Challenge 3

Build a **REST API** to integrate your model into a web app, making it a breeze for the team to use. Keep it developer-friendly – not everyone speaks 'data scientist'! 
Your API should support two use cases:
1. Predict the value of a diamond.
2. Given the features of a diamond, return n samples from the training dataset with the same cut, color, and clarity, and the most similar weight.

### Challenge 4

Observability is key. Save every request and response made to the APIs to a **proper database**.

---

## How to run
🦎

### Project's Notion Page: Planned Tasks, In-Progress Tasks, Completed Tasks
https://maratonacheckpoint.notion.site/Xtream-AI-e392e154df754c9293409079608e5fe3?pvs=4

### Setting Up the Environment

1. **Clone the Repository**:
   ```sh
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a Virtual Environment**:
   ```sh
   python -m venv venv
   ```

3. **Activate the Virtual Environment**:
   ```sh
   source venv/bin/activate
   ```

4. **Install the Required Packages**:
   ```sh
   pip install -r requirements.txt
   ```

### Running the Machine Learning Pipeline

1. **Run the Main Pipeline**:
   ```sh
   python scripts/main.py --data_url data/diamonds.csv --model_type linear
   ```
   Replace `linear` with `xgboost` to train the XGBoost model.

   This script will:
   - Load and preprocess the data.
   - Train the specified model.
   - Evaluate the model.
   - Save the model and its performance metrics.

### Running the API Server

1. **Start the Flask API Server**:
   ```sh
   python scripts/api.py
   ```
   The server will start running on `http://127.0.0.1:5000`.

2. **API Endpoints**:
   - **Predict Diamond Price**:
     - URL: `http://127.0.0.1:5000/predict`
     - Method: `POST`
     - Request JSON format:
       ```json
       {
         "carat": 0.5,
         "cut": "Premium",
         "color": "E",
         "clarity": "VS2",
         "depth": 61.5,
         "table": 55.0,
         "x": 5.1,
         "y": 5.1,
         "z": 3.15
       }
       ```
   - **Get Similar Diamonds**:
     - URL: `http://127.0.0.1:5000/get_similar_diamonds`
     - Method: `POST`
     - Request JSON format:
       ```json
       {
         "cut": "Premium",
         "color": "E",
         "clarity": "VS2",
         "weight": 0.5,
         "n": 5
       }
       ```

### Running the Streamlit App

1. **Start the Streamlit App**:
   ```sh
   streamlit run streamlit.py
   ```
   The app will open in your default web browser.

2. **Using the Streamlit App**:
   - The app has two sections:
     - **Price Predictor**: Enter the diamond features and click "Predict" to get the estimated price.
     - **Similar Diamonds Finder**: Enter the diamond features and the number of similar diamonds to find, then click "Find Similar Diamonds" to get the results.

### Additional Notes

- Ensure the Flask API server is running before using the Streamlit app.
- All API requests and responses are logged in the SQLite database `api_requests.db` for observability.
- The model and its performance metrics are saved in the `models` directory, and the best model is stored in `models/best_model`.

If you encounter any issues or need further assistance, please refer to the documentation or contact the project maintainers.
