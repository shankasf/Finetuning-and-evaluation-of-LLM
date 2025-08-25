import requests
from bs4 import BeautifulSoup

# URL 
url = "https://github.com/karanpratapsingh/system-design"

# Send a request to fetch the HTML content of the webpage
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all relevant text content (adjust based on how the webpage is structured)
    paragraphs = soup.find_all('p')
    
    # Open a file to save the extracted content
    with open("raw_data.txt", "w", encoding='utf-8') as f:
        for para in paragraphs:
            # Write each paragraph into the file
            f.write(para.get_text(separator=" ").strip() + "\n\n")
            
    print("Text data successfully extracted and saved raw_data.txt")
else:
    print("Failed to retrieve the webpage.")
