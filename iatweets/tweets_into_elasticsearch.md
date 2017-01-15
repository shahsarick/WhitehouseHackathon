This probably isn't even worth turning into a real Python script, I started the Python interpreter and did this freehand inside a GNU Screen session.  This brief snippet of code takes the archived tweets (which have been deserialized and loaded into a Python dict) and stores them into a local instance of Elasticsearch:


```
import json
import requests

count = 1
for i in parsed_tweets:
    for j in i['tweets']:
        r = requests.put('http://localhost:9200/candidate_tweets/tweet/' + str(count), data=json.dumps(j))
        count = count + 1
```

If it's worth it, I'll turn it into a real utility.

