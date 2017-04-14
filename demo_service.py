import flask
import functools
import flask_api
import json
import traceback

import Utils
import demo_exceptions

app = flask.Flask('demo')
logger = Utils.init_logger("demo")


def product_service_exception_handler(func):
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
def split_words():
    data = flask.request.data


if __name__ == "__main__":
    app.run()