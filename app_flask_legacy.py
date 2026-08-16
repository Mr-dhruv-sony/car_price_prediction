from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize the Flask App
app = Flask(__name__)

# Load the trained Random Forest model
model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))

@app.route('/')
def home():
    # Renders the HTML UI
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # 1. Extract numerical features from the form
        present_price = float(request.form['Present_Price'])
        kms_driven = int(request.form['Kms_Driven'])
        owner = int(request.form['Owner'])
        years_since_manufacture = int(request.form['Years_Since_Manufacture'])
        
        # 2. Extract and encode categorical features (Fuel Type)
        fuel_type = request.form['Fuel_Type']
        Fuel_Type_Petrol = 1 if fuel_type == 'Petrol' else 0
        Fuel_Type_Diesel = 1 if fuel_type == 'Diesel' else 0
            
        # 3. Extract and encode categorical features (Seller Type)
        seller_type = request.form['Seller_Type']
        Seller_Type_Individual = 1 if seller_type == 'Individual' else 0
            
        # 4. Extract and encode categorical features (Transmission)
        transmission = request.form['Transmission']
        Transmission_Manual = 1 if transmission == 'Manual' else 0
        
        # 5. Make the prediction using the model
        final_features = [np.array([present_price, kms_driven, owner, years_since_manufacture, 
                                    Fuel_Type_Diesel, Fuel_Type_Petrol, Seller_Type_Individual, Transmission_Manual])]
        prediction = model.predict(final_features)
        
        # Format the output
        output = round(prediction, 2)
        
        # Return the result to the webpage
        return render_template('index.html', prediction_text=f'Estimated Selling Price: ₹ {output} Lakhs')

if __name__ == "__main__":
    app.run(debug=True)