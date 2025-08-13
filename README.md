# image-checker-suite-demo


# Env Setup for local runs
python3 -m venv photoenv
source photoenv/bin/activate


# Install dependencies
pip install -r requirements.txt


python exposure.py "/home/zach/Desktop/PhotoData"

# run (locally)
streamlit run app.py


# Run Tests locally
 python3 -m unittest discover -s tests

 
 # Test real image groupings. This function will group into large number of folders. 
 # This is the most recent function that groups similar images.
 python /home/zach/Documents/dev/image-checker-suite-demo/evaluate_groupsing.py
