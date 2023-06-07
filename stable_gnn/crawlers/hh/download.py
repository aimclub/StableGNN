import json
import os
import time
import logging
import requests

logger = logging.getLogger(__name__)


def get_page(page_num=0):
    """
    Creating a method to get a page with a list of vacancies.
    Args:
        page_num - The index of the page starts from 0. The default value is 0, i.e. the first page

    Returns:

    """

    # Reference for GET request parameters
    text = "NAME:Data science"
    params = {
        "text": text,  # Filter text. The name should contain the word "Analyst"
        "area": 1,  # The search is carried out by vacancies in the city of Moscow
        "page": page_num,  # Index of the search page on HH
        "per_page": 100,  # Number of vacancies on 1 page
        "period": 30,
        "label": "not_from_agency",
        "order_by": "publication_time",
    }

    req = requests.get("https://api.hh.ru/vacancies", params)

    # We decode his answer so that the Cyrillic alphabet is displayed correctly

    data = (
        req.content.decode()
    )

    req.close()
    return data


# Reading the first 2000 vacancies
for page in range(0, 20):
    # Converting the text of the request response to the Python reference
    jsObj = json.loads(get_page(page))

    # Save the files to the folder {path to the current document with the script}\docs\pagination
    # We determine the number of files in the folder to save the document with the request response
    # We use the received value to form the document name

    nextFileName = "./docs/pagination/{}.json".format(
        len(os.listdir("./docs/pagination"))
    )

    # Create a new document, write the request response to it, then close it
    f = open(nextFileName, mode="w", encoding="utf8")
    f.write(json.dumps(jsObj, ensure_ascii=False, sort_keys=True, indent=2))
    f.close()

    # Check to the last page if there are fewer than 2000 vacancies
    if (jsObj["pages"] - page) <= 1:
        break

    # An optional delay, but in order not to load hh services, we will leave it. 5 sec we can wait
    time.sleep(0.25)

logger.info("The search pages are collected")
