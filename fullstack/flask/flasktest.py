from flask import Flask, render_template
from werkzeug.utils import escape

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template("homepage.html")

@app.route("/about")
def about():
    text = '<div style="background-color:green;">This</div>'
    text += '<div style="background-color:cyan;">is a different</div>'
    text += '<div style="background-color:hotpink;">page!</div>'
    return text


# username = "cooluser"
@app.route('/user/<username>')
def show_user_profile(username):
    return f'Username: {escape(username)}'

# for i in range(0,10):
#     newuser = username + str(i)
#     @app.route('/user/newuser')
#     def show_new_user(newuser):
#         return f'Username: {escape(newuser)}'

@app.route('/number/<int:a_number>')
def show_number(a_number):
    return (f"WOW IT's NUMBER {a_number}")



if __name__ == "__main__":
    app.run(debug=True)