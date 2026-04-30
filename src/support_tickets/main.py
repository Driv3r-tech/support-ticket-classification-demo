import argparse
from readers import csv_process, json_process
import pathlib

parser = argparse.ArgumentParser(description="Обработка файлов")
parser.add_argument('--input', dest='input_path', default="data/requests.json", type=str)
parser.add_argument('--output', dest='output_path', default="output.json", type=str)

args = parser.parse_args()

file_input = pathlib.Path(args.input_path)

if file_input.is_file():
    if args.input_path.endswith('.csv'):
        csv_process(args)
    elif args.input_path.endswith('.json'):
        json_process(args)
    else:
        print("Необходим файл для загрузки форматом: .csv или .json")
else:
    print("Файл для загрузки не найден!")







