import json
import os
import time
import logging
import requests

logger = logging.getLogger(__name__)

# We get a list of previously created files with a list of vacancies and go through it in a cycle
for fl in os.listdir("./docs/pagination"):
    # Open the file, read its contents, close the file
    f = open("./docs/pagination/{}".format(fl), encoding="utf8")
    json_text = f.read()
    f.close()

    # Convert the resulting text into a directory object
    json_obj = json.loads(json_text)

    # We receive and go through the list of vacancies directly
    for v in json_obj["items"]:
        # We turn to the API and get detailed information on a specific vacancy
        req = requests.get(v["url"])
        data = req.content.decode()
        req.close()

        logger.info(v["id"])

        # Creating a file in json format with the job ID as the name
        # Write the request response to it and close the file

        fileName = "./docs/vacancies/{}.json".format(v["id"])
        f = open(fileName, mode="w", encoding="utf8")

        jsonVocObj = json.loads(data)
        f.write(json.dumps(jsonVocObj, ensure_ascii=False, sort_keys=True, indent=2))
        f.close()

        time.sleep(0.25)

logger.info("Vacancies are collected")
