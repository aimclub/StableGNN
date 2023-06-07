from enum import Enum


class NameCase(Enum):
    nom = "nom"  # nominative
    gen = "gen"  # genitive
    dat = "dat"  # dative
    acc = "acc"  # accusative
    ins = "ins"  # ablative
    abl = "abl"  # prepositional


class Fields(Enum):
    # photo_id = "photo_id"
    # verified = "verified"
    sex = "sex"
    bdate = "bdate"
    # city = "city"
    # country = "country"
    # home_town = "home_town"
    # has_photo = "has_photo"
    # photo_50 = "photo_50"
    # photo_100 = "photo_100"
    # photo_200_orig = "photo_200_orig"
    # photo_200 = "photo_200"
    # photo_400_orig = "photo_400_orig"
    # photo_max = "photo_max"
    # photo_max_orig = "photo_max_orig"
    # online = "online"
    domain = "domain"
    # has_mobile = "has_mobile"
    # contacts = "contacts"
    # site = "site"
    education = "education"
    universities = "universities"
    schools = "schools"
    status = "status"
    # last_seen = "last_seen"
    counters = "counters"
    # common_count = "common_count"  # not used for unauthorized users
    # occupation = "occupation"
    nickname = "nickname"
    # relatives = "relatives"
    # relation = "relation"
    # personal = "personal"
    # connections = "connections"
    # exports = "exports"
    # activities = "activities"
    # interests = "interests"
    # music = "music"
    # movies = "movies"
    # tv = "tv"
    # books = "books"
    # games = "games"
    about = "about"
    # quotes = "quotes"
    # can_post = "can_post"
    # can_see_all_posts = "can_see_all_posts"
    # can_see_audio = "can_see_audio"
    # can_write_private_message = "can_write_private_message"
    # can_send_friend_request = "can_send_friend_request"
    # is_favorite = "is_favorite"
    # is_hidden_from_feed = "is_hidden_from_feed"
    timezone = "timezone"
    screen_name = "screen_name"
    maiden_name = "maiden_name"
    # crop_photo = "crop_photo"
    # is_friend = "is_friend"
    # friend_status = "friend_status"
    career = "career"
    military = "military"
    # blacklisted = "blacklisted"
    # blacklisted_by_me = "blacklisted_by_me"
    # can_be_invited_group = "can_be_invited_group"


def all_fields():
    return [e.value for e in Fields]
