#!/usr/bin/env python

from flask import Flask, url_for, send_from_directory, request

app = Flask(__name__)

@app.route('/', methods=['GET'])
def test():
	return "hello\n"


if __name__ == '__main__':

    app.run(debug=True, use_reloader=False)
