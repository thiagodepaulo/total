#!/usr/bin/python

import sys
import urllib
import re
import json

from bs4 import BeautifulSoup

import socket
socket.setdefaulttimeout(10)

cache = {}

i = 0
for line in open(sys.argv[1]):
    
    fields = line.rstrip('\n').split('\t')
    sid = fields[0]
    uid = fields[1]
    tweet = None
    
    text = "Not Available"
    if sid in cache:
        text = cache[sid]
        print(i," ",text)
        i=i+1
    else:
        try:
            f = urllib.urlopen("http://twitter.com/%s/status/%s" % (uid, sid))
            html = f.read().replace("</html>", "") + "</html>"
            soup = BeautifulSoup(html)
            jstt = soup.find_all("p","js-tweet-text")
            tweets = list(set([x.get_text() for x in jstt]))
            if(len(tweets)) > 1:
                print(i," Maior ",text)
                continue
            text = tweets[0]
            print("Tweet ",text)
            cache[sid] = tweets[0]
            for j in soup.find_all("input", "json-data", id="init-data"):
                js = json.loads(j['value'])
                if(js.has_key("embedData")):
                    tweet = js["embedData"]["status"]
                    text  = js["embedData"]["status"]["text"]
                    cache[sid] = text
                    break
        except Exception:
            continue

    if(tweet != None and tweet["id_str"] != sid):
        text = "Not Available"
        cache[sid] = "Not Available"
        
    text = text.replace('\n', ' ',)
    text = re.sub(r'\s+', ' ', text)
#    print(json.dumps(tweet, indent=2))
    print("\t",fields,[text],encode('utf-8'))
    i=i+1
            
