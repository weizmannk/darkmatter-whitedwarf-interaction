The criteria that I used are the following:
- > 3 days between first and last detection
- < 60 days between the first and last alert
- >= 5 detections
- for real/bogus, I used rb > 0.9 before June 2019; I used drb > 0.65 and braai > 0.65 after June 2019
- for the galaxy crossmatch, I considered galaxies between 20Mpc and 200Mpc and projected distances between 10kpc and 130kpc
- for the light curves, I selected programid=3 according to DR17, while programid=1 and 2 are for the whole dataset. We can get full light curves if you prefer, I made the selection to avoid possible troubles as a starter.

There are >~ 6k candidates in total. It should be fairly easy to make extra cuts (e.g., on absolute mag) by combining light curve and xmatch information. Is there any other operation you would like me to help with?




https://github.com/growth-astro/ztfrest/blob/master/forced.py#L82

to read in the files and make quick plots. It will need some small
modifications but should get you close.

I took a random example and https://fritz.science/source/ZTF20aauvmix
is classified as a Ia, for example.

For this task, I suggest writing a loop over sources and querying for
classifications with something like this:

api_token = ""
host = "https://fritz.science/"

endpoint = "sources/ZTF23aaaykdg"
url = urllib.parse.urljoin(host, f'/api/{endpoint}')
headers = {'Authorization': f'token {api_token}'}

r = requests.get(url, headers=headers)
classifications = r.json()['data']['classifications']





--name ZTF18aachiql --ra  145.267154142845 --declination 4.7329642016285  --date-start 2459682.7025926 --date-end 2459710.7196181  --channel 
