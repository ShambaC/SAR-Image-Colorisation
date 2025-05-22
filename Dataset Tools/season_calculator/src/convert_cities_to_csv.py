import os
from bs4 import BeautifulSoup
import csv
import traceback

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
if not script_dir:
    script_dir = os.getcwd()

# Path to the HTML file
html_file_path = os.path.join(script_dir, 'cities.htm')

# Path for the output CSV file
csv_file_path = os.path.join(script_dir, 'cities.csv')

try:
    # Read the HTML file
    print(f"Reading HTML file from: {html_file_path}")
    with open(html_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
        
    print(f"HTML file successfully read, size: {len(html_content)} bytes")

    # Parse HTML using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    print("HTML parsed with BeautifulSoup")

    # Find the table in the HTML
    table = soup.find('table')

    # Check if the table was found
    if not table:
        print("Table not found in the HTML file.")
        exit(1)

    print("Table found in HTML")

    # Find all rows in the table
    rows = table.find_all('tr')
    print(f"Found {len(rows)} rows in the table")

    # Define CSV header
    header = [
        "Country/Territory", 
        "Capital",
        "Largest City", 
        "Second Largest City"
    ]

    # Extract data from the table
    data = []

    # Skip the first 4 rows which are header rows
    for i, row in enumerate(rows[4:], start=5):  # Start counting from 5 for better error reporting
        try:
            cells = row.find_all('td')
            
            if not cells:
                continue  # Skip rows without data cells
            
            # Get country/territory
            country = ""
            if cells[0].find('a'):
                country = cells[0].find('a').get_text(strip=True)
            else:
                country = cells[0].get_text(strip=True)
            
            # Get capital, largest city, and second largest city
            capital = ""
            largest_city = ""
            second_largest = ""
            
            # Different patterns in the table
            if len(cells) >= 2:
                capital_cell = cells[1]
                
                # Extract capital
                if capital_cell.find('a'):
                    capital = capital_cell.find('a').get_text(strip=True)
                else:
                    capital = capital_cell.get_text(strip=True)
                
                # Handle different row patterns
                if len(cells) == 2:
                    # Case where capital is in first cell, second cell has largest and second largest
                    if capital_cell.has_attr('colspan') and capital_cell['colspan'] == '2':
                        largest_city = capital  # Capital is also the largest city
                elif len(cells) >= 3:
                    # Case with separate columns for capital, largest, and second largest
                    if capital_cell.has_attr('colspan') and capital_cell['colspan'] == '2':
                        largest_city = capital  # Capital is also the largest city
                        if len(cells) >= 3:  # There's a third cell
                            if cells[2].find('a'):
                                second_largest = cells[2].find('a').get_text(strip=True)
                            else:
                                second_largest = cells[2].get_text(strip=True)
                    else:
                        # Normal case: separate capital, largest city, second largest
                        if cells[2].find('a'):
                            largest_city = cells[2].find('a').get_text(strip=True)
                        else:
                            largest_city = cells[2].get_text(strip=True)
                        
                        if len(cells) >= 4:
                            if cells[3].find('a'):
                                second_largest = cells[3].find('a').get_text(strip=True)
                            else:
                                second_largest = cells[3].get_text(strip=True)
            
            # Create row data
            row_data = [country, capital, largest_city, second_largest]
            data.append(row_data)
            
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            traceback.print_exc()
            continue

    print(f"Extracted data from {len(data)} rows")

    # Write the data to a CSV file
    with open(csv_file_path, 'w', encoding='utf-8', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write the header
        csv_writer.writerow(header)
        
        # Write the data
        csv_writer.writerows(data)

    print(f"Conversion completed. CSV file saved at: {csv_file_path}")

except Exception as e:
    print(f"An error occurred: {e}")
    traceback.print_exc()
