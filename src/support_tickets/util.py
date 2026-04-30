from request import Request

def request_from_dict(data: dict) -> Request:
    required_fields = ["id", "category", "priority", "user", "status", "message"]

    for field in required_fields:
        if field not in data:
            raise ValueError(f"Отсутствует обязательное поле: {field}")

    return Request(
        request_id=data["id"],
        category=data["category"],
        priority=data["priority"],
        user=data["user"],
        status=data["status"],
        message=data["message"]
    )

def parse_line(line):
    line = line.strip()

    if not line:
        return None

    blocks = line.split(";")
    d = {}

    for block in blocks:
        parts = block.split("=", 1)

        if len(parts) != 2:
            raise ValueError(f"Некорректный блок: {block}")

        key = parts[0].strip()
        value = parts[1].strip()
        d[key] = value

    required_fields = ["id", "category", "priority", "user", "status", "message"]

    for field in required_fields:
        if field not in d:
            raise ValueError(f"Отсутствует обязательное поле: {field}")

    return Request(
        request_id=d["id"],
        category=d["category"],
        priority=d["priority"],
        user=d["user"],
        status=d["status"],
        message=d["message"]
    )


def build_statistics(records):
    statistics = {
        "total_requests": len(records),
        "by_category": {},
        "by_priority": {}
    }

    for record in records:
        category = record.category
        priority = record.priority

        if category is not None:
            statistics["by_category"][category] = statistics["by_category"].get(category, 0) + 1

        if priority is not None:
            statistics["by_priority"][priority] = statistics["by_priority"].get(priority, 0) + 1

    return statistics