
from flask import Flask, render_template, redirect, request

import voice_recognition


app = Flask(__name__)



@app.route('/')
def hi():
	return render_template("./index.html")

@app.route('/',methods = ['POST'])
def song():
	if request.method == 'POST':

		name = request.values['name']
		#print(name)
		voice_recognition.recorder(name)

		music = voice_recognition.display(name)
		
	return render_template("./index.html",tables=[music.to_html(classes='table table-striped table-bordered', header="true",index="true")])

if __name__ == '__main__':
	#app.debug = True
	app.run(debug = True)
#