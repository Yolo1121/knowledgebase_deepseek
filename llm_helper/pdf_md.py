# -*- coding: utf-8 -*-
# @Time    : 2024/7/10 10:49
# @Author  : zhanghaoxiang
# @File    : pdf_md.py
# @Software: PyCharm
import mammoth
import markdownify
import re
from bs4 import BeautifulSoup, NavigableString, Tag
import bs4
import copy
from pdf2docx import Converter
import time, datetime
import shutil
import os
import tempfile
class HTMLCompleter():
    def __init__(self, html: str):
        self.soup = BeautifulSoup(html, "html.parser")

    def _complete_table_row(self):
        # 查找包含跨行合并的单元格
        merge_cells = self.soup.find_all(attrs={"rowspan": True})
        # 对每个需要拆分的单元格进行处理
        for cell in merge_cells:
            # 获取单元格的行列号和跨行数
            parent_cell = cell.parent
            row_index = parent_cell.index(cell)
            n_span = int(cell.get("rowspan"))
            del cell['rowspan']
            roll_sibling = parent_cell
            # 将单元格拆分成多个单元格
            for _ in range(1, n_span):
                insert_cell = copy.copy(cell)
                roll_sibling = roll_sibling.next_sibling
                roll_sibling.insert(row_index, insert_cell)
        return self

    def _complete_table_col(self):
        merge_cells = self.soup.find_all(attrs={"colspan": True})
        for cell in merge_cells:
            n_span = int(cell['colspan'])
            col_index = cell.parent.index(cell)
            del cell['colspan']
            for _ in range(1, n_span):
                insert_cell = copy.copy(cell)
                cell.parent.insert(col_index, insert_cell)

        return self

    def complete_table(self):
        self._complete_table_col()._complete_table_row()
        return self

    def _strip_tags(self, soup, invalid_tags: list = ['strong', ]):
        """
        delete invalid_tags but keep the content of the them.
        """
        for tag in soup.findAll(True):
            # print(tag)
            if tag.name in invalid_tags:
                s = ""
                for c in tag.contents:
                    if isinstance(c, NavigableString):
                        s += c
                    else:
                        c = self._strip_tags(c, invalid_tags)
                        if c is not None and hasattr(c, "contents") and len(c.contents) > 0:
                            s += c.contents[0]
                tag.replaceWith(s)
        return soup

    def unnest_table(self):
        '''
        Un-nest table.
        '''
        for table in self.soup.find_all('table', recursive=False):
            sub_tables = table.find_all('table')
            for sub_table in sub_tables:
                s = ""
                for el in sub_table.find_all(True):
                    for c in el.contents:
                        if isinstance(c, NavigableString):
                            s += c.string.strip()
                sub_table.replaceWith(s)
        return self

    def strip_tags(self, invalid_tags: list = ['strong', ]):
        """
        delete invalid_tags but keep the content of the them.
        """
        self._strip_tags(self.soup)
        return self

    def table_head_fix(self, strip_tags=None):
        for table in self.soup.find_all('table'):
            tr = table.find('tr')
            for th in tr.find_all('td'):
                th.name = 'th'
            if strip_tags:
                self._strip_tags(table, invalid_tags=strip_tags)
        return self

    def clap_table(self, n_rows=2):
        """
        Convert a table with only a few rows into a paragraph.
        The table with few rows is likely due to an error during PDF conversion.
        """
        assert n_rows >= 1 and isinstance(n_rows,
                                          int), "param: n_rows should be an integer, and greater than or equal to 1."
        for table in self.soup.find_all('table', recursive=False):
            rows = table.find_all('tr')
            if len(rows) <= n_rows:
                new_tag = self.soup.new_tag('p')
                tr_list = []
                for row in rows:
                    c_list = []
                    for ele in row.find_all(True):
                        for c in ele.contents:
                            if isinstance(c, NavigableString):
                                c_list.append(c.string.strip())
                    if c_list:
                        tr_list.append(' '.join(c_list))
                for tr in tr_list:
                    _ = self.soup.new_tag('p')
                    _.string = tr.strip()
                    new_tag.append(_)
                table.replaceWith(new_tag)
        return self

    def _del_tags(self, tags=['img']):
        for tag in tags:
            for ele in self.soup.find_all(tag):
                ele.decompose()

        return self

    def output_soup(self):
        return self.soup

    def output_html(self):
        return str(self.soup)

    def _merge_first_tr_line(self):
        """
        When table header with merged cells, split them and keep only last header line.

        Returns:
            bs4.element.Tag: _description_
        """
        table_soups = self.soup.find_all('table')
        ## del all <img> nested in the table.
        for table_soup in table_soups:
            for tag in ['img', ]:
                for ele in table_soup.find_all(tag):
                    ele.decompose()

        # Only search `rowspan` in first <tr> tag, for it's the header of the whole table.
        for table_soup in table_soups:

            merged_cell = table_soup.find('tr').find_all(attrs={'rowspan': True})
            max_rowspan_in_first_tr_line = 0
            for mc in merged_cell:
                tmp = int(mc.get('rowspan'))
                max_rowspan_in_first_tr_line = tmp if max_rowspan_in_first_tr_line < tmp else max_rowspan_in_first_tr_line
            if max_rowspan_in_first_tr_line > 1:
                tag_tr = table_soup.find_all('tr')
                ## Seperate cell if colspan
                for tr in tag_tr[:max_rowspan_in_first_tr_line]:
                    colspan = tr.find_all(attrs={"colspan": True})
                    for cs in colspan:
                        n_cs = int(cs.get('colspan'))
                        col_index = cs.parent.index(cs)
                        del cs['colspan']
                        for _ in range(1, n_cs):
                            insert_cell = copy.copy(cs)
                            cs.parent.insert(col_index, insert_cell)

                ## Seprate cell if rowspan
                # merge_cells = merge_cells
                for cell in merged_cell:
                    parent = cell.parent
                    tr_index = parent.index(cell)
                    n_span = int(cell.get("rowspan"))
                    del cell['rowspan']
                    roll_sibling = parent

                    # 将单元格拆分成多个单元格
                    for _ in range(1, n_span):
                        insert_cell = copy.copy(cell)
                        roll_sibling = roll_sibling.next_sibling
                        roll_sibling.insert(tr_index, insert_cell)

                ## merge header value, Keep only the last filled table header as the header of unique one.
                for tr in tag_tr[:max_rowspan_in_first_tr_line - 1]:
                    _prev, _next = tr, tr.next_sibling
                    for _p, _n in zip(_prev, _next):
                        if _p.text != _n.text:
                            _n.string = _p.text + _n.text
                    _prev.decompose()

        return self

    @classmethod
    def _merge_table_with_same_rownum(cls, previous_sibling, next_sibling):
        """
        Merge 2 tables on the row when they are in same row number.
        Args:
            previous_sibling (bs4.element.Tag): the first table
            next_sibling (bs4.element.Tag): the second table

        Returns:
            bs4.element.Tag: merged table.
        """
        tr_in_previous = previous_sibling.find_all('tr')
        tr_in_next = next_sibling.find_all('tr')
        assert len(tr_in_previous) == len(tr_in_next), "Two tables must with same row numbers when they merge"
        for _ptr, _ntr in zip(tr_in_previous, tr_in_next):
            for _td in _ntr.children:
                _ = copy.copy(_td)
                _ptr.append(_)
        return previous_sibling

    def merge_adjacent_tables_with_same_rownum(self):
        """
        Merge adjacent 2 tables if they are with same shape
        """
        self._merge_first_tr_line()
        children = self.soup.children
        merge_flag = False
        for child in children:
            if merge_flag:
                continue
            _prev, _next = child, child.next_sibling
            if isinstance(_prev, Tag) and isinstance(_next, Tag) and (_prev.name == 'table') and (
                    _next.name == "table"):
                _prev_tr_num = len(_prev.find_all('tr'))
                _next_tr_num = len(_next.find_all('tr'))
                if _prev_tr_num == _next_tr_num:
                    child.replaceWith(self._merge_table_with_same_rownum(_prev, _next))

                    _next.decompose()
                    merge_flag = True
                else:
                    merge_flag = False
            else:
                merge_flag = False
        return self

# def strip_tags(html: str, invalid_tags: list):
#     """
#     Delete invalid_tags but keep the content of the them.
#     """
#     soup = BeautifulSoup(html, 'html.parser')
#     for tag in soup.findAll(True):
#         if tag.name in invalid_tags:
#             s = ""
#             for c in tag.contents:
#                 if not isinstance(c, NavigableString):
#                     c = strip_tags(str(c), invalid_tags)
#                 s += str(c)
#             tag.replaceWith(s)
#     return soup


# def firstline2head(html):
#     soup = BeautifulSoup(html, 'html.parser')
#     for table in soup.find_all('table'):
#         tr = table.find('tr')
#         for th in tr.find_all('td'):
#             th.name = 'th'
#     # soup = strip_tags(str(soup), ['strong',])
#     th_p = soup.select('th > p')

#     return soup.__str__()

def doc2md(doc_path, md_path, html_path=None):
    with open(doc_path, 'rb') as f:
        html = mammoth.convert_to_html(f).value
    html = HTMLCompleter(html).table_head_fix(strip_tags=['strong', ]).complete_table().output_html()
    md = markdownify.markdownify(
        html,
        heading_style="ATX",
        strip=['img', 'a'],
        newline_style='SPACE',
        escape_asterisks=True,
        escape_underscores=False  # 转义下划线
    )
    pat = re.compile('\n{3,}')
    md = pat.sub('\n', md)
    with open(md_path, "w") as md_file:
        md_file.write(md)

    if html_path:
        with open(html_path, 'w') as html_file:
            html_file.write(html)
    return None

def docx2md(f):
    """
    f: Binary file object of MS .docx
    """
    html = mammoth.convert_to_html(f).value
    html = HTMLCompleter(html).table_head_fix(strip_tags=['strong', ]).complete_table().output_html()
    md = markdownify.markdownify(
        html,
        heading_style="ATX",
        strip=['img', 'a'],
        newline_style='SPACE',
        escape_asterisks=True,
        escape_underscores=False  # 转义下划线
    )
    pat = re.compile('\n{3,}')
    md = pat.sub('\n', md)
    return md
def pdf2md(f, md_path: str = None, html_path: str = None, docx_path: str = None, start=0, end=None, pages=None):
    base_path = tempfile.mkdtemp()
    pdf_file = os.path.join(base_path, f.name.split('\\')[-1])

    with open(pdf_file, 'wb') as pf:
        pf.write(f.read())
    docx_file = os.path.join(base_path, 'out_put.docx')
    converter = Converter(pdf_file)
    converter.convert(docx_file, start=start, end=end, pages=pages, multi_processing=False, cpu_count=0)

    converter.close()

    html = mammoth.convert_to_html(docx_file).value
    html = HTMLCompleter(html).unnest_table().clap_table().merge_adjacent_tables_with_same_rownum().table_head_fix(
        strip_tags=['strong', ]).complete_table().output_html()

    if docx_path:
        shutil.copy2(docx_file, docx_path)

    md = markdownify.markdownify(
        html,
        heading_style="ATX",
        strip=['img', 'a'],
        newline_style='SPACE',
        escape_asterisks=True,
        escape_underscores=False  # 转义下划线
    )
    pat = re.compile('\n{3,}')
    md = pat.sub('\n', md)
    if md_path:
        with open(md_path, "w", encoding='utf8') as md_file:
            md_file.write(md)

    if html_path:
        with open(html_path, 'w',encoding='utf8') as html_file:
            html_file.write(html)
    # print(base_path)
    # print('-' * 100)
    # shutil.rmtree(base_path)
    return md







