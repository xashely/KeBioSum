
import re
import json

with open('links.txt','r') as f:
    links = f.readlines()

metadata_links = []
pdf_parses = []
for line in links:
    if line.startswith('wget'):
        if 'metadata' in line:
            matches = re.findall(r"https://.*(?=')",line)
            if matches:
                for match in matches:
                    metadata_links.append(match)
        elif 'pdf_parses' in line:
            matches = re.findall(r"https://.*(?=')",line)
            if matches:
                for match in matches:
                    pdf_parses.append(match)

download_links = []
for idx, link in enumerate(metadata_links):
    download_links.append({"metadata": link, "pdf_parses":pdf_parses[idx]})

with open('download_links.txt','w') as f:
    json.dump(download_links,f)