import openpyxl
import csv
import sys

# open given workbook
# and store in excel object
excel = openpyxl.load_workbook(sys.argv[1])

# select the active sheet
sheet = excel.active

# writer object is created
col = csv.writer(open(sys.argv[1].rstrip(".xlsx") + ".csv",
                      'w',
                      newline=""))

# writing the data in csv file
for r in sheet.rows:
    # row by row write
    # operation is perform
    col.writerow([cell.value for cell in r])