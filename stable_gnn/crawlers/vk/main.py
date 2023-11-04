import json
import re
import sys
from time import sleep
import logging

from app.models import Group, GroupTagPivot, Tag, User, UserGroupPivot, UserUserPivot
from app.user.user import get_user_api_fabric

logger = logging.getLogger(__name__)


def user_get_or_create(user_obj):
    """
    Get user or create if not exists then return.

    Args:
        user_obj: User object

    Returns:

    """

    # Try to find user

    user_exists = User.select().where(User.id_external == user_obj["id"]).exists()

    # Insert if not exists

    if not user_exists:
        user_query = User(id_external=user_obj["id"], content=user_obj)
        user_query.save()

    # Return created user

    return User.select().where(User.id_external == user_obj["id"]).get()


def group_get_or_create(group_obj):
    """
    Get group or create if not exists then return.
    
    Args:
        group_obj: 

    Returns:

    """

    # Try to find group

    group_exists = Group.select().where(Group.id_external == group_obj["id"]).exists()

    # Insert if not exists

    if not group_exists:
        group_query = Group(id_external=group_obj["id"], content=group_obj)
        group_query.save()

    # Return created group

    return Group.select().where(Group.id_external == group_obj["id"]).get()


def connect_user_to_group(user_id, group_id):
    """
    Create edge between user and group.

    Args:
        user_id: User ID
        group_id: Group ID

    Returns:

    """

    # Get connection from user to group

    user_group_exists = (
        UserGroupPivot.select()
        .where(UserGroupPivot.user_id == user_id, UserGroupPivot.group_id == group_id)
        .exists()
    )

    # If not exists, then create a new one

    if not user_group_exists:
        user_group_query = UserGroupPivot(user_id=user_id, group_id=group_id)
        user_group_query.save()

    # Return created connection

    return (
        UserGroupPivot.select()
        .where(UserGroupPivot.user_id == user_id, UserGroupPivot.group_id == group_id)
        .get()
    )


def connect_user_to_user(user_id, another_user_id):
    """
    Get connection from user to group.

    Args:
        user_id: User ID
        another_user_id: Another User ID

    Returns:

    """
    user_user_exists = (
        UserUserPivot.select()
        .where(UserUserPivot.from_id == user_id, UserUserPivot.to_id == another_user_id)
        .exists()
    )

    # If not exists, then create a new one

    if not user_user_exists:
        UserUserPivot(from_id=user_id, to_id=another_user_id).save()

    # Return created connection

    return (
        UserUserPivot.select()
        .where(UserUserPivot.from_id == user_id, UserUserPivot.to_id == another_user_id)
        .get()
    )


def main():
    """
    Read username from ARGV.

    Returns:

    """
    user_name = ""
    try:
        user_name = sys.argv[1]
    except IndexError:
        logger.info("./main.py username")
        exit(1)

    # Get object via API
    # Get details about user

    user_api = get_user_api_fabric()
    user_id = user_api.get_user_id_by_verbose_name(user_name)

    # Info about user

    user = user_api.get_info(user_id)

    # Save user to database

    user_obj = user_get_or_create(user[0])
    groups = user_api.get_groups(user_id, True)
    for group in groups:
        groupObj = group_get_or_create(group)
        connect_user_to_group(user_obj.id, groupObj.id)

    # Get list of user's friends

    friends = user_api.get_friends(user_id)
    for friend in friends:
        friend_obj = user_get_or_create(friend)
        connect_user_to_user(user_obj.id, friend_obj.id)

        # Exclude closed profiles

        if "is_closed" in friend and friend["is_closed"]:
            continue
        if "deactivated" in friend and (
            friend["deactivated"] == "banned" or friend["deactivated"] == "deleted"
        ):
            continue

        friend_groups = user_api.get_groups(friend["id"], True)

        for friend_group in friend_groups:
            friend_group_obj = group_get_or_create(friend_group)
            connect_user_to_group(friend_obj.id, friend_group_obj.id)
            sleep(0.05)

    # Get list of user's followers

    followers = user_api.get_followers(user_id)
    for follower in followers:
        follower_obj = user_get_or_create(follower)
        connect_user_to_user(user_obj.id, follower_obj.id)

        # Exclude closed profiles

        if "is_closed" in follower and follower["is_closed"]:
            continue

        if "deactivated" in follower and (
            follower["deactivated"] == "banned" or follower["deactivated"] == "deleted"
        ):
            continue

        follower_groups = user_api.get_groups(follower["id"], True)

        for follower_group in follower_groups:
            follower_group_obj = group_get_or_create(follower_group)
            connect_user_to_group(follower_obj.id, follower_group_obj.id)

    # Connect groups to tags
    # Read aliases

    aliases = json.loads(open("./aliases.json", "r").read())

    # Create connections

    for group in groups:
        group_id = group["id"]
        group_name = group["name"].lower().strip()
        group_name = " ".join(group_name.split())
        if group_name == "" or "анекдотов.net" in group_name:
            continue

        # Connect tags to group

        tags = []
        for key, value in aliases.items():
            for string in value:
                if re.compile(r"\b({0})\b".format(string), re.IGNORECASE).search(
                    group_name
                ):
                    if key not in tags:
                        tags.append(key)

        # Skip groups without tags

        if not tags:
            continue

        # Always JS

        if "vue.js" in tags and "javascript" not in tags:
            tags.append("javascript")
        if "react.js" in tags and "javascript" not in tags:
            tags.append("javascript")
        if "angular.js" in tags and "javascript" not in tags:
            tags.append("javascript")
        if "node.js" in tags and "javascript" not in tags:
            tags.append("javascript")
        if "nuxt.js" in tags and "javascript" not in tags:
            tags.append("javascript")
        if "nest.js" in tags and "node.js" not in tags:
            tags.append("javascript")
        if "nest.js" in tags and "javascript" not in tags:
            tags.append("javascript")
        if "next.js" in tags and "node.js" not in tags:
            tags.append("javascript")

        logger.info(group_name, tags)

        # Get group from list

        group_db = Group.select().where(Group.id_external == group_id).get()

        # Connect existing tags to group

        for tag in tags:
            tag_single_query = Tag.select().where(Tag.name == tag)
            if tag_single_query.exists():
                tagObj = Tag.select().where(Tag.name == tag).get()
                group_tag_obj = (
                    GroupTagPivot()
                    .select()
                    .where(
                        GroupTagPivot.group_id == group_db.id,
                        GroupTagPivot.tag_id == tagObj.id,
                    )
                )
                if not group_tag_obj.exists():
                    group_tag_query = GroupTagPivot(
                        group_id=group_db.id, tag_id=tagObj.id
                    )
                    group_tag_query.save()


if __name__ == "__main__":
    main()
