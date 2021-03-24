# https://deepnote.com/project/webscraping-pair-programming-yZLBey_jQIOHaMhck-ea0w/%2Fnotebook.ipynb#00001-5a664854-c744-47d9-a248-1dc03cb70d05

#First off, grap the webpage
webpage = requests.get("https://forecast.weather.gov/MapClick.php?lat=37.777120000000025&lon=-122.41963999999996#.YFsm0v6nxH5")
#webpage.content

# Parse the webpage into something usable
soup = BeautifulSoup(webpage.content, "html.parser")
# # Split it up
# html = list(soup.children)[2]
# body = list(html.children)[3]

# This locates some P tags which contain all the information we need
data = soup.find_all("p")
# Remove anything we don't need
fields = data[8:]
#print(fields)

print("==========")
daysplit = []

# Build a list of all the things we need
for i in range(0, len(data), 4):
    daysplit.append(fields[i:i+4])

# Remove some unnecessary data
del daysplit[9:]

# Key: 0:period - 1:description - 2:short-desc - 3. Temp
# Output, this is what an item list looks like
for i in range(0, len(daysplit[0])):
    print(daysplit[0][i])

nice_data = []

seperator = ('.')

# Strip all the tags from the data
for report in daysplit:
    #print("=====I AM A REPORT")
    # print(day)
    for attribute in report:
        #replace('<p class="period-name">', "")
        #print("=====I AM AN ATTRIBUTE")
        #print(attribute)
        attribute = str(attribute)
        attribute = attribute.split(seperator, 1)[0]
        #print(type(attribute))
        attribute = attribute.replace('</p>',"")
        attribute = attribute.replace('<p class="period-name">', "")
        attribute = attribute.replace('<br/><br/></p>', "")
        attribute = attribute.replace('<p><img alt="', "")
        attribute = attribute.replace('<p class="short-desc">', "")

        attribute = attribute.replace('<p class="temp temp-high">', "")
        attribute = attribute.replace('<p class="temp temp-low">', "")
        attribute = attribute.replace('<br/><br/>', "")
       # attribute = attribute.replace('."', "")
        attribute = attribute.replace('<br/>', " ")


        #print(attribute)
        nice_data.append(attribute)


print(nice_data[0:4])
database = []

for i in range(0, len(nice_data), 4):
    #print(nice_data[i:i+4])
    database.append(nice_data[i:i+4])

print(database[0][2])

df = pd.DataFrame(database)

print(df)
