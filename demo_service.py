# coding=utf-8

import flask
import functools
import flask_api
import json
import traceback

import Utils
import demo_exceptions
import TextExtractor
import Predictor
import Mes

app = flask.Flask('demo')
logger = Utils.init_logger("demo")
cutter = TextExtractor.WordCutter()
predictor = Predictor.Predictor(docs=None, trainable=False)


def service_exception_handler(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        content = {
            "msg": "",
            "data": {},
            "url": "",
            "error": {"type": "", "message": "", "message_chs": ""}
        }
        headers = {"content-type": "application/json"}
        try:
            return func(*args, **kw)
        except (AssertionError, ValueError, TypeError) as e:
            content["error"]["type"] = "FormatError"
            content["error"]["message"] = "The input format is invalid. EXCEPTION: {}".format(e)
            content["error"]["message_chs"] = "格式或值无效"
            logger.error(func.__name__ + " failed, MESSAGE: " + content["error"]["message"] +
                         "TRACEBACK: " + traceback.format_exc())
            return json.dumps(content), flask_api.status.HTTP_400_BAD_REQUEST, headers
        except demo_exceptions.DEMOBaseException as e:
            content["error"]["type"] = e.error_code
            content["error"]["message"] = e.message
            content["error"]["message_chs"] = e.ch_message
            logger.error(func.__name__ + " failed, MESSAGE: " + content["error"]["message"] +
                         "TRACEBACK: " + traceback.format_exc())
            return json.dumps(content), e.status_code, headers
        except KeyError as e:
            content["error"]["type"] = "InvalidInput"
            content["error"]["message"] = "The request param error, a required parameter was missing. " \
                                          "Exception: {}".format(e)
            content["error"]["message_chs"] = "键缺失"
            logger.error(func.__name__ + " failed, MESSAGE: " + content["error"]["message"] +
                         "TRACEBACK: " + traceback.format_exc())
            return json.dumps(content), flask_api.status.HTTP_400_BAD_REQUEST, headers
        except Exception as e:
            content["error"]["type"] = "InternalServerError"
            content["error"]["message"] = "Internal Server Error. EXCEPTION: " + str(e)
            content["error"]["message_chs"] = "服务器内部错误"
            logger.error(func.__name__ + " failed, MESSAGE: " + content["error"]["message"] +
                         "TRACEBACK: " + traceback.format_exc())
            return json.dumps(content), flask_api.status.HTTP_500_INTERNAL_SERVER_ERROR, headers
    return wrapper


@app.route("/split_words/", methods=["POST"])
@service_exception_handler
def split_words():
    content = {"msg": "", "data": {}, "url": "",
               "error": {"type": "", "message": "", "message_chs": ""}}
    headers = {"content-type": "application/json"}
    text = flask.request.form["text"]
    text = unicode(text)
    word = cutter.split(text)
    content["data"]["words"] = word
    return json.dumps(content), flask_api.status.HTTP_200_OK, headers


@app.route("/predict/", methods=["POST"])
@service_exception_handler
def prediction():
    content = {"msg": "", "data": {}, "url": "",
               "error": {"type": "", "message": "", "message_chs": ""}}
    headers = {"content-type": "application/json"}
    text = flask.request.form["words"]
    words = json.loads(text)
    logits = predictor.predict(words)
    rare_words = []
    for word in words:
        if word[2] != word[0]:
            rare_words.append(word[0])
    content["data"]["logits"] = logits.tolist()
    content["data"]["rare_words"] = ','.join(rare_words)
    return json.dumps(content), flask_api.status.HTTP_200_OK, headers


@app.route("/", methods=["GET"])
@service_exception_handler
def index():
    return flask.render_template('index.html')

if __name__ == "__main__":
    app.run(host="localhost", debug=True, port=Mes.DEMO_API_PORT)
