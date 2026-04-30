import json
from util import parse_line, build_statistics, request_from_dict
from pprint import pprint

def csv_process(args):
    records = []
    with open(args.input_path, encoding="UTF-8") as file_in:
        for line in file_in:
            try:
                record = parse_line(line)
                if record is not None:
                    records.append(record)
            except ValueError as e:
                print(f"Ошибка в записи {line}: {e}")

    # print(records)
    print(build_statistics(records))


def json_process(args):
    records = []
    with open(args.input_path, encoding="UTF-8") as file_in:
        raw_records = json.load(file_in)

    for item in raw_records:
        try:
            record = request_from_dict(item)
            records.append(record)
        except ValueError as e:
            print(f"Ошибка в записи {item}: {e}")

    # print(records)
    pprint(build_statistics(records))

