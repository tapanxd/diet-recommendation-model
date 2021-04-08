from flask import Flask
from diet_Copy import *
import csv23
app = Flask(__name__)
@app.route('/')
def predict():
    return "Demo page"

@app.route('/<string:a>/<string:w>/<string:h>/<string:p>')
def func2(a, w, h, p):
    pref = None
    if p == "Veg":
        pref = 0
    elif p == "Non-Veg":
        pref = 1

    break_fast = []
    lunch = []
    dinner = []
    diet = main_func(a, w, pref, h)
    for food in diet:
        with open("food.csv") as f:
            reader = csv23.reader(f)
            for row in reader:
                if(row[0] == food):
                    if row[1] == "1":
                        break_fast.append(row[0] + " " + row[5] + " " + "cal")
                        
                    if row[2] == "1":
                        lunch.append(row[0]+ " " + row[5] + " " + "cal")
                        
                    if row[3] == "1":
                        dinner.append(row[0]+ " " + row[5] + " " + "cal")
                    
    # print(break_fast,lunch,dinner)
    brstr = "<br>".join(break_fast)
    lustr = "<br>".join(lunch)
    distr = "<br>".join(dinner)
    
    return "<b>Breakfast: </b><br>" + brstr + "<br><b>Lunch: </b><br>" + lustr + "<br><b>Dinner: </b><br>" + distr
    

if __name__ == "__main__":
    app.run(debug = True)
