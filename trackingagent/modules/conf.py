# import the necessary packages
from json_minify import json_minify
import json
import modules.log as l

class Conf:
    def __init__(self, confPath):

        logit = l.Log()
        # load and store the configuration and update the
        # object's dictionary
        logit.info(f"Loading config from {confPath}")
        conf = json.loads(json_minify(open(confPath).read()))
        self.__dict__.update(conf)

    def __getitem__(self, k):
        # return the value associated with the supplied key
        return self.__dict__.get(k, None)
