# -*- coding: utf-8 -*-
# @Time    : 2024/8/1 11:19
# @Author  : zhanghaoxiang
# @File    : table_rec.py
# @Software: PyCharm
import os
import cv2
from paddleocr import PPStructure,save_structure_res
from bs4 import BeautifulSoup

def html_to_txt1(html_str):
    soup = BeautifulSoup(html_str, 'html.parser')
    tables = soup.find("table")
    header_row = tables.find("tr")
    header_cols = header_row.find_all("td")
    # 最大跨行数
    maxCellrow = 0
    # 每一列的跨行数
    lenSplit = []
    dic = {}
    # 找到最大的分行数
    for index, cell in enumerate(header_cols):
        splittxt = cell.get_text().split("\n")
        dic[index] = splittxt
        lenSplit.append(len(splittxt))
        if len(splittxt) > maxCellrow:
            maxCellrow = len(splittxt)
    # 创建新行
    for index in range(maxCellrow - 1):
        new_row = soup.new_tag('tr')
        for index in range(0, len(header_cols)):
            new_cell = soup.new_tag('td', align="center")  # 创建新的单元格，居中显示
            new_cell.string = ""
            new_row.append(new_cell)
            header_row.insert_after(new_row)
    # 合并修改
    rows = tables.find_all("tr")[:(maxCellrow)]
    row_span = 1
    for row_index, row in enumerate(rows):
        for col_index, cell in enumerate(row.find_all("td")):
            if (row_index + 1) > lenSplit[col_index]:
                cell.string = ""
            else:
                cell.string = dic[col_index][row_index]
                cell.attrs["align"] = "center"
    '''
        for row_index, row in enumerate(rows):
        next_trs=row.next_siblings
        tds = row.find_all('td')
        for next_tr in next_trs:
            if tds[0].get_text() == next_tr.contents[0].get_text():
                row_span += 1
                tds[0]["rowspan"] = row_span
                next_tr.contents[0].extract()
            if next_tr.contents[0].get_text()=="":
                row_span += 1
                tds[0]["rowspan"] = row_span
            row_span = 1


    '''

    formatted_html = soup.prettify()
    return formatted_html
table_engine = PPStructure(layout=False, show_log=True)

save_folder = './output'
img_path = "D:\\temp\\zhongqi1\\zhongqi1_8_res\\parse_result\\8_1_table.jpg"
img = cv2.imread(img_path)
result = table_engine(img)
save_structure_res(result,save_folder, os.path.basename(img_path).split('.')[0])

for line in result:
    line.pop('img')
    print(line)
print(html_to_txt1(result[0]["res"]["html"]))