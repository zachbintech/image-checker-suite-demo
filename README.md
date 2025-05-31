# image-checker-suite-demo


# Env Setup for local runs
python3 -m venv photoenv
source photoenv/bin/activate


# Install dependencies
pip install -r requirements.txt


# run (locally)
streamlit run app.py


# Run Tests locally
 python3 -m unittest discover -s tests
