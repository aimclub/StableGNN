import csv
import glob
import html
import json
import os
import re
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Datetime object containing current date and time

now = datetime.now()

# Load JSON files

excludes = json.loads(open("./excludes.json", "r").read())
aliases = json.loads(open("./aliases.json", "r").read())


def strip_tags(value):
    """Returns the given HTML with all tags stripped."""
    tag_re = re.compile(r"(<!--.*?-->|<[^>]*>)")
    # Remove well-formed tags, fixing mistakes by legitimate users
    no_tags = tag_re.sub("", value)
    # Clean up anything else by escaping
    return html.escape(no_tags)


# Open CSV file for writing in append mode

suffix = now.strftime("%Y-%m-%d_%H:%M")
csvfile = open("./docs/csv/output_" + suffix + ".csv", "a")
filewriter = csv.writer(
    csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
)
filewriter.writerow(["text", "keys"])

# Get list of all scrapped vacancies

vacancies = glob.glob("./docs/vacancies/*")
vacanciesCount = len(vacancies)
i = 0
for fl in vacancies:
    # Increment counter

    i = i + 1

    # Open file

    f = open(fl, encoding="utf8")
    json_text = f.read()
    f.close()

    # Read JSON to array format

    json_arr = json.loads(json_text)

    # If key_skills list is empty, then skip this step

    if not json_arr["key_skills"]:
        continue

    # Extract only required fields

    id = json_arr["id"]
    name = json_arr["name"]
    description = strip_tags(json_arr["description"]).lower()
    key_skills = [skill.get("name") for skill in json_arr["key_skills"]]

    # Pandas, Numpy, Matplotlib / Plotly

    tmp_key_skills = []
    for skill in key_skills:
        if "," in skill:
            tmp_key_skills + skill.split(",")
        else:
            tmp_key_skills.append(skill)

    # Sanitize list

    filter_key_skills = []
    for skill in tmp_key_skills:
        _skill = skill.lower().strip().rstrip(".")
        # Let's skip excludes

        if _skill in excludes:
            continue

        # Check if skill is not an alias

        filter_skills = _skill
        for key, value in aliases.items():
            if _skill in value:
                filter_skills = key.lower()

        # Add to output array

        filter_key_skills.append(filter_skills)

    # Sort and remove duplicates

    filter_key_skills = sorted(set(filter_key_skills))

    # Parse skill, we need keys which exists in text

    skills = []
    for skill in filter_key_skills:
        # Check if skill from keys is in main text
        if re.search(
            r"" + re.escape(skill),
            name + " " + description,
            re.MULTILINE | re.IGNORECASE,
        ):
            skills.append(skill)

    # If key_skills list is empty, then skip this step

    if not skills:
        continue

    # Always JS

    if "vue.js" in skills and "javascript" not in skills:
        skills.append("javascript")
    if "react.js" in skills and "javascript" not in skills:
        skills.append("javascript")
    if "angular.js" in skills and "javascript" not in skills:
        skills.append("javascript")
    if "node.js" in skills and "javascript" not in skills:
        skills.append("javascript")
    if "nuxt.js" in skills and "javascript" not in skills:
        skills.append("javascript")
    if "nest.js" in skills and "node.js" not in skills:
        skills.append("javascript")
    if "nest.js" in skills and "javascript" not in skills:
        skills.append("javascript")
    if "next.js" in skills and "node.js" not in skills:
        skills.append("javascript")

    # Build string of skills

    skills = sorted(skills)
    skills = ",".join(skills)

    # Status report

    message = "[{}/{}] id:{} skills:{}".format(i, vacanciesCount, id, skills)
    
    logger.info(message)

    # Put data to file

    filewriter.writerow([name + " " + description, skills])
