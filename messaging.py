import pywhatkit as kit
import json
import threading

# import urllib.request, urllib.parse, urllib.error
# import ssl

data = '''
{
  "Obama" : "+919999999999",
  "Biden" : "+918888888888"
}'''

#replace with API call

# # Ignore SSL certificate errors
# ctx = ssl.create_default_context()
# ctx.check_hostname = False
# ctx.verify_mode = ssl.CERT_NONE

# url = ""

# uh = urllib.request.urlopen(url, context=ctx).read().decode()

# number = json.loads(uh)

number = json.loads(data)

def sendmessage(name,msg,hr,mint):
  try:
    t = threading.Thread(target=kit.sendwhatmsg,args=(number[name],msg,hr,mint))
    t.daemon = True
    t.start()
    return
  except(KeyError):
    print(f"No Registered Number for {name}")
    return

