from flask import Flask, render_template
import os

# os.environ["FLASK_APP"] = r"Tic_tac_toe\\UI\\Flask_trial.py"
app = Flask(__name__)

posts = [
    {
        'author': 'Pradyumna M K Prasad',
        'title': 'Blog post 1',
        'content': 'First post content',
        'date_posted': '29th Dec 2019'
    },
    {
        'author': 'Donnatella Xavier',
        'title': 'Blog post 2',
        'content': 'Second post content',
        'date_posted': '29th Dec 2019'
    }
]

@app.route('/')
@app.route('/home')
def home():
    return render_template('buttons_template.html', posts=posts)

@app.route('/about')
def about():
    return render_template('about.html', title='About')


if __name__ == '__main__':
    app.run(debug=True)