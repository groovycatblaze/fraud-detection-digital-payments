# transaction fraud detection in digital payments

inspired by real-world challenges in financial technology; trying to detect fraudulent activities in real-time while providing **explainability and accessibility**, ensuring secure, transparent, and inclusive digital commerce

## features
- **model**: machine learning pipeline trained on synthetic transaction data.  
- **explainability**: integrated SHAP and Grad-CAM visualizations to explain fraud predictions.  
- **UI**: flask-based application with HTML/CSS/JS frontend, responsive design, and accessibility (WCAG 2.0).  
- **scalability**: SQL database with support for distributed storage (can extend to AWS).  
- **sub-second inference**: optimized models to detect suspicious transactions in real-time.  

## tech stack
- **languages**: Python, C++, JavaScript, SQL  
- **libraries**: PyTorch, Pandas, NumPy, SHAP, Scikit-learn, Flask  
- **frontend**: HTML, CSS, JavaScript (responsive design)  
- **backend**: Flask REST APIs, SQLAlchemy ORM  
- **cloud ready**: AWS S3/EC2/Lambda integration possible  

## usage
```bash
# clone repo
git clone https://github.com/groovycatblaze/fraud-detection-digital-payments.git
cd fraud-detection-digital-payments

# install dependencies
pip install -r requirements.txt

# Run app
python app.py
