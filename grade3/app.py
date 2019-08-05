import flask

import pandas as pd


from sklearn.externals import joblib

regressor = joblib.load('model/model.pkl')


app = flask.Flask(__name__, template_folder='templates')

@app.route('/grade3', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
        
        sex = flask.request.form['sex']
        age = flask.request.form['age']
        studytime = flask.request.form['studytime']
        failures = flask.request.form['failures']
        freetime = flask.request.form['freetime']
        absences = flask.request.form['absences']
        goout = flask.request.form['goout']
        first = flask.request.form['first']
        second = flask.request.form['second']
        
        


        input_variables = pd.DataFrame([[sex,age,studytime,failures,freetime,absences,goout,first,second]],
                                       columns=['sex', 'age', 'studytime','failures','freetime','absences','goout','first','second'],
                                       dtype=float,
                                       index=['input'])

        prediction = regressor.predict(input_variables)[0]
        
    
        return flask.render_template('main1.html',
                                   
                        
                                     result=prediction
                                     )

if __name__ == '__main__':
    app.debug = True
    app.run()
