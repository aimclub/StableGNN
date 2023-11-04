-- List of users
create table users
(
    id          serial primary key not null,
    id_external text               not null,
    content     json      default '{}'::json not null,
    created_at  timestamp default now()
);

create unique index users_id_uindex
    on users (id);

create index users_id_external_index
    on users (id_external);

-- From user to user
create table user_user
(
    from_id    integer not null,
    to_id      integer not null,
    created_at timestamp default now()
);

-- List of groups
create table groups
(
    id          serial primary key not null,
    id_external text               not null,
    content     json      default '{}'::json not null,
    created_at  timestamp default now()
);

create unique index groups_id_uindex
    on groups (id);

create index groups_id_external_index
    on groups (id_external);

-- Connection between user and group
create table user_group
(
    user_id     integer not null,
    group_id    integer not null,
    posts_count integer,
    likes_count integer,
    created_at  timestamp default now()
);

-- List of tags for marking groups
create table tags
(
    id         serial
        constraint tags_pk primary key,
    name       text    not null,
    is_valid   boolean not null default TRUE,
    created_at timestamp        default now()
);

create unique index tags_id_uindex
    on tags (id);

-- Connection between group and tags
create table group_tag
(
    group_id   integer not null,
    tag_id     integer not null,
    created_at timestamp default now()
);
