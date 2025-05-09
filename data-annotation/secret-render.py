import requests
from bs4 import BeautifulSoup

def find_max_value_in_tuple_list(list, position_comparison):
    max_value = float('-inf')  # Initialize with negative infinity
    for sublist in list:
        coordinate = int(sublist[position_comparison])
        if coordinate > max_value:
            max_value = coordinate
    return max_value


def parse_html_table(html_doc):
    soup = BeautifulSoup(html_doc, 'html.parser')
    table = soup.find('table')

    if table is None:
        return None

    table_data = []
    rows = table.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        if not cols:
            cols = row.find_all('th')
        row_data = [col.text.strip() for col in cols]
        table_data.append(row_data)
    return table_data[1:]

def create_grid(table):
    max_x = find_max_value_in_tuple_list(table, 0)
    max_y = find_max_value_in_tuple_list(table, 2)
    matrix = [[" " for _ in range(max_x+1)] for _ in range(max_y+1)]
    for row in table:
        matrix[max_y-int(row[2])][int(row[0])] = row[1]
    return matrix


def render_secret_message(google_doc_url):
    response = requests.get(google_doc_url)
    table = parse_html_table(response.content)
    matrix = create_grid(table)
    for row in matrix:
        print(" ".join(map(str, row)))


render_secret_message(
    "https://docs.google.com/document/d/e/2PACX-1vRMx5YQlZNa3ra8dYYxmv-QIQ3YJe8tbI3kqcuC7lQiZm-CSEznKfN_HYNSpoXcZIV3Y_O3YoUB1ecq/pub")

print("Test")
render_secret_message("https://docs.google.com/document/d/e/2PACX-1vQGUck9HIFCyezsrBSnmENk5ieJuYwpt7YHYEzeNJkIb9OSDdx-ov2nRNReKQyey-cwJOoEKUhLmN9z/pub")
