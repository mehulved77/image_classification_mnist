import argparse
parser = argparse.ArgumentParser()
parser.add_argument("word", help="Print the word in upper case letters")
args = parser.parse_args()
print(args.word.upper())
