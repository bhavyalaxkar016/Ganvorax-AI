GANVORAX AI - Multi-page Streamlit app

Files:
- app.py
- requirements.txt

Deploy update:
1. Replace the existing app.py in your GitHub repository with this app.py.
2. Keep requirements.txt in the repository.
3. Commit the changes to the main branch.
4. Streamlit Community Cloud should detect the commit and redeploy automatically.
5. Refresh your app URL after deployment completes.

Run locally:
pip install -r requirements.txt
streamlit run app.py

This version retains the original TF-IDF + cosine similarity approach and the six illustrative job records.
It adds sidebar navigation and session state so analysis results remain available while navigating pages.
