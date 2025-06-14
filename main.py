#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as ps
import tensorflow as tfs
from flask import Flask
app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello, SmartNinja!"


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)

