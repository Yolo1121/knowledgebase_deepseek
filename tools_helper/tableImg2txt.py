# -*- coding: utf-8 -*-
# @Time    : 2024/7/25 11:12
# @Author  : zhanghaoxiang, 表格恢复的原码库进行了修改，增加图像增强方式，使用中使用修改后的文件
# @File    : tableImg2txt.py
# @Software: PyCharm
from bs4 import BeautifulSoup
from lxml import html
from lxml.html import builder as E
from bs4.element import Tag
from collections import defaultdict
def html_beautify(input_str):
    soup = BeautifulSoup(input_str, 'html.parser')

    # 处理表头跨行的情况

    # 添加CSS样式

    style = soup.new_tag('style')
    style.string = '''
    table {
        table-layout: fixed;
        width: 100%;
    }
    tr, td {
        word-wrap: break-word;
        white-space: normal;
    }
    '''
    soup.body.append(style)

    # 格式化HTML字符串
    formatted_html = soup.prettify()
    return formatted_html
def html_to_txt(html_str):
    soup = BeautifulSoup(html_str, 'html.parser')

    # 处理表头
    header_row = soup.find('tr')
    header_cells = header_row.find_all('td')
    cell_len = len(header_cells)
    curInex = 0
    rowList=[]
    for cell in header_cells:
        srtList=cell.get_text().split("\n")
        if len(srtList)==0:
            curInex+=1
            continue
        cell.string=srtList[0]
        cell.attrs["align"]='center'
        #初始行号
        rowlistindex = 0
        for str in srtList[1:]:
            if len(rowList)==0 or len(rowList)<(len(srtList)-1):
                new_row = soup.new_tag('tr')
                for index in range(0, curInex):
                    new_cell = soup.new_tag('td', align="center")  # 创建新的单元格，居中显示
                    new_cell.string = ""
                    new_row.append(new_cell)
                new_cell = soup.new_tag('td', align="center", )  # 创建新的单元格，居中显示
                new_cell.string = str
                new_row.append(new_cell)
                for index in range(curInex + 1, cell_len):
                    new_cell = soup.new_tag('td', align="center")  # 创建新的单元格，居中显示
                    new_cell.string = ""
                    new_row.append(new_cell)
                rowList.append(new_row)
            else:
                #特定列进行文本添
                new_row = rowList[rowlistindex]
                colindex=0
                for col in new_row.find_all("td"):
                    if colindex==curInex:
                        col.string=str
                        colindex += 1
                    else:
                        colindex+=1
                rowlistindex += 1
        curInex += 1

    for row in rowList:
        header_row.insert_after(row)
    # 处理表格内容,采用相同的方式进行
    data_rows = soup.find_all('tr')[(len(rowList) + 1):]

    for row in data_rows:
        datarowslist = []
        curdatacolInex = 0
        cyrdatarowIndex = 0
        cells = row.find_all('td')
        for cell in cells:
            srtList = cell.get_text().split("\n")
            if len(srtList) == 0:
                curdatacolInex += 1
                continue
            cell.string = srtList[0]
            cell.attrs["align"] = 'center'
            for str in srtList[1:]:
                if len(datarowslist) == 0 or len(datarowslist) < (len(srtList) - 1):
                    new_row = soup.new_tag('tr')
                    for index in range(0, curdatacolInex):
                        new_cell = soup.new_tag('td', align="center")  # 创建新的单元格，居中显示
                        new_cell.string = ""
                        new_row.append(new_cell)
                    new_cell = soup.new_tag('td', align="center", )  # 创建新的单元格，居中显示
                    new_cell.string = str
                    new_row.append(new_cell)
                    for index in range(curdatacolInex + 1, cell_len):
                        new_cell = soup.new_tag('td', align="center")  # 创建新的单元格，居中显示
                        new_cell.string = ""
                        new_row.append(new_cell)
                    datarowslist.append(new_row)
                else:
                    # 特定列进行文本添
                    new_row = datarowslist[cyrdatarowIndex]
                    colindex = 0
                    for col in new_row.find_all("td"):
                        if colindex == curdatacolInex:
                            col.string = str
                            colindex += 1
                        else:
                            colindex += 1
                    cyrdatarowIndex += 1
            curdatacolInex += 1
        for row in rowList:
            header_row.insert_after(row)

    # 格式化HTML字符串
    formatted_html = soup.prettify()
    return formatted_html
from wired_table_rec import WiredTableRecognition
from PIL import Image,ImageOps

def html_to_txt1(html_str):
    soup = BeautifulSoup(html_str, 'html.parser')
    tables = soup.find("table")
    header_row = tables.find("tr")
    header_cols= header_row.find_all("td")
    #最大跨行数
    maxCellrow=0
    #每一列的跨行数
    lenSplit=[]
    dic={}
    #找到最大的分行数
    for index, cell in enumerate(header_cols):
        splittxt=cell.get_text().split("\n")
        dic[index]=splittxt
        lenSplit.append(len(splittxt))
        if len(splittxt) > maxCellrow:
            maxCellrow=len(splittxt)
    #创建新行
    for index in range(maxCellrow-1):
        new_row = soup.new_tag('tr')
        for index in range(0, len(header_cols)):
            new_cell = soup.new_tag('td', align="center")  # 创建新的单元格，居中显示
            new_cell.string = ""
            new_row.append(new_cell)
            header_row.insert_after(new_row)
    # 合并修改
    rows=tables.find_all("tr")[:(maxCellrow)]
    row_span = 1
    for row_index, row in enumerate(rows):
        for col_index,cell in enumerate(row.find_all("td")):
            if (row_index+1)>lenSplit[col_index]:
                cell.string=""
            else:
                cell.string=dic[col_index][row_index]
                cell.attrs["align"]="center"
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











table_rec = WiredTableRecognition()
img_path ="D:\\temp\\zhongqi1\\zhongqi1_8_res\\parse_result\\8_1_table.jpg"
table_str, elapse = table_rec(img_path)

res=html_to_txt1(table_str)
print(table_str)
print(elapse)
print(res)
