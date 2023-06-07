import sys
import logging

from gns.crawlers.vk.app.models import Edge, Top

logger = logging.getLogger(__name__)


def main():
    # Read username from ARGV
    user_name = ""
    try:
        user_name = sys.argv[1]
    except IndexError:
        logger.info("./get_graph.py username")
        exit(1)

    # Get root object
    user_db = (
        Top.select()
        .where(
            (
                (Top.id_external == user_name)
                | (Top.content["nickname"] == user_name)
                | (Top.content["domain"] == user_name)
            ),
            Top.source == "vk",
            Top.type == "user",
        )
        .join(Top="t2", on=(Top.id == Edge.from_id))
    )

    # Get relations
    logger.info(user_db)
    exit(1)


if __name__ == "__main__":
    main()
