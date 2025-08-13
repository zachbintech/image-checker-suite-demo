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
 python evaluate_groupsing.py


 # This is the function that currently runs to detect artifacting in images. It currently only looks at Red Channels. Need to add looking at green and blue.
 python artifacts/bad_bands.py 


