from flask import Flask
from diet_Copy import *
app = Flask(__name__)
@app.route('/')
def predict():
    return "Demo page"

# @app.route('/<string:a>/<string:w>/<string:h>/<string:p>')
# def func2(a, w, h, p):
#     pref = None
#     if p == "Veg":
#         pref = 0
#     elif p == "Non-Veg":
#         pref = 1

#     diet = main_func(a, w, pref, h)
#     dietstr = "<br>".join(diet)
#     return dietstr

if name == "main":
    app.run(debug = True)
