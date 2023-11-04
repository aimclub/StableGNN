from datetime import datetime

from peewee import *
from playhouse.postgres_ext import *

db = PostgresqlDatabase(
    "gnn", user="postgres", password="rootpass", host="postgres", port=5432
)


class User(Model):
    id = IntegerField(primary_key=True)
    id_external = TextField()
    content = JSONField()
    created_at = DateTimeField(default=datetime.now)

    class Meta:
        database = db
        db_table = "users"


class UserUserPivot(Model):
    from_id = IntegerField()
    to_id = IntegerField()
    created_at = DateTimeField(default=datetime.now)

    class Meta:
        database = db
        db_table = "user_user"
        primary_key = False


class Group(Model):
    id = IntegerField(primary_key=True)
    id_external = TextField()
    content = JSONField()
    created_at = DateTimeField(default=datetime.now)

    class Meta:
        database = db
        db_table = "groups"


class UserGroupPivot(Model):
    user_id = IntegerField()
    group_id = IntegerField()
    # posts_count = IntegerField()
    # likes_count = IntegerField()
    created_at = DateTimeField(default=datetime.now)

    class Meta:
        database = db
        db_table = "user_group"
        primary_key = False


class Tag(Model):
    id = IntegerField(primary_key=True)
    name = TextField()
    is_valid = BooleanField(default=True)
    created_at = DateTimeField(default=datetime.now)

    class Meta:
        database = db
        db_table = "tags"


class GroupTagPivot(Model):
    group_id = IntegerField()
    tag_id = IntegerField()
    created_at = DateTimeField(default=datetime.now)

    class Meta:
        database = db
        db_table = "group_tag"
        primary_key = False
