# Toxicity Detector

The Toxicity Detector is an AI-powered solution for detecting and 
classifying toxic content, including fake news and biased statements. It 
utilizes advanced natural language processing techniques, specifically a 
fine-tuned RoBERTa transformer-based model, to accurately identify and 
categorize toxic sentences.

## Features

- FastAPI-based API: Provides endpoints for making toxicity classification 
predictions using the trained RoBERTa model.
- Transformer-based Model: Utilizes the RoBERTa model, a state-of-the-art 
transformer architecture, for accurate and reliable toxicity detection.
- Streamlit User Interface: Offers a user-friendly interface for 
interacting with the toxicity detector, allowing users to input sentences 
and receive toxicity predictions in real-time.

## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/your-username/toxicity-detector.git
   ```

2. Install the dependencies:

   ```shell
   pip install -r requirements.txt
   ```

3. Run the FastAPI server:

   ```shell
   uvicorn fast:app --host 0.0.0.0 --port 8000
   ```

4. Run the Streamlit user interface:

   ```shell
   streamlit run streamlit.py
   ```

5. Open your browser and visit `http://localhost:8501` to access the 
Toxicity Detector user interface.

## Usage

1. Use the Streamlit user interface to select a sentence from the dropdown 
menu.
2. Click the "Predict" button to obtain the toxicity classification 
prediction for the selected sentence.
3. The user interface will display the predicted toxicity category, such 
as "No Bias," "Hate Speech," "Fake News," "Political Bias," "Racial Bias," 
or "Gender Bias."

## Contributing

Contributions to the Toxicity Detector project are welcome! If you 
encounter any issues or have suggestions for improvements, please feel 
free to open an issue or submit a pull request. Remember to follow the 
project's code of conduct.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

We would like to express our gratitude to the developers and contributors 
of the FastAPI, Hugging Face Transformers, and Streamlit libraries, whose 
work has made this project possible.

## Contact

For questions or inquiries, please contact 
[Ancastal](mailto:ancastal@outlook.it).
