#!/usr/bin/env python
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
# from __future__ import unicode_literals
import sys
import json
import logging
import argparse

ARG_DEFAULTS = {'log':sys.stderr, 'llevel':logging.ERROR}
DESCRIPTION = """Prints all tweets as a list of JSON objects.
If multiple WARC files are given, prints a list of them, as JSON of this format:
[
  {
    "path":"path/to/file1.warc", "tweets":[{tweet1..},{tweet2...}]
  }
  {
    "path":"path/to/file2.warc", "tweets":[{tweet1..},{tweet2...}]
  }
]
"""


def main(argv):

  parser = argparse.ArgumentParser(description=DESCRIPTION)
  parser.set_defaults(**ARG_DEFAULTS)

  parser.add_argument('warcs', metavar='path/to/record.warc', nargs='+',
    help='Un-gzipped WARC files.')
  parser.add_argument('-l', '--list', action='store_true',
    help='Just print a list of tweets as independent JSON objects, one per line.')
  parser.add_argument('-L', '--log', type=argparse.FileType('w'),
    help='Print log messages to this file instead of to stderr. Warning: Will overwrite the file.')
  parser.add_argument('-q', '--quiet', dest='llevel', action='store_const', const=logging.CRITICAL)
  parser.add_argument('-v', '--verbose', dest='llevel', action='store_const', const=logging.INFO)
  parser.add_argument('-D', '--debug', dest='log_level', action='store_const', const=logging.DEBUG)

  args = parser.parse_args(argv[1:])

  logging.basicConfig(stream=args.log, level=args.llevel, format='%(message)s')
  tone_down_logger()

  tweet_files = []
  for path in args.warcs:
    tweets = list(parse_warc(path))
    tweet_files.append({'path':path, 'tweets':tweets})

  if args.list:
    for tweets in tweet_files:
      for tweet in tweets['tweets']:
        json.dump(tweet, sys.stdout)
        print()
  else:
    if len(tweet_files) == 1:
      json.dump(tweet_files[0]['tweets'], sys.stdout)
    else:
      json.dump(tweet_files, sys.stdout)


def parse_warc(warc_path):
  """Usage:
  import parse_warc
  for tweet in parse_warc.parse_warc('path/to/filename.warc'):
    # "tweet" is a JSON object.
    print tweet.location
  """
  tweet_json = ''
  header = False
  with open(warc_path, 'rU') as warc:
    for line in warc:
      if header:
        if line.startswith('Content-Length:'):
          header = False
        continue
      else:
        if line == 'WARC/1.0\n':
          header = True
          if tweet_json:
            tweet = json.loads(tweet_json)
            yield tweet
          tweet_json = ''
          continue
      tweet_json += line


def tone_down_logger():
  """Change the logging level names from all-caps to capitalized lowercase.
  E.g. "WARNING" -> "Warning" (turn down the volume a bit in your log files)"""
  for level in (logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG):
    level_name = logging.getLevelName(level)
    logging.addLevelName(level, level_name.capitalize())


def fail(message):
  sys.stderr.write(message+"\n")
  sys.exit(1)


if __name__ == '__main__':
  sys.exit(main(sys.argv))
