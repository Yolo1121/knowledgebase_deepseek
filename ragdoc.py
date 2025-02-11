import shutil
import fitz

class RagDoc:
    def __init__(self, docjson):
        self.docpath = docjson['docpath']
        self.docformat = docjson['docformat'] #doc,docx,pdf,xls,xlsx
        self.docid = docjson['docid']
    def downloadfile(self, savedir):
        if 'hdfs:' in self.docpath:
            pass
        else:
            shutil.copy(self.docpath, savedir)
    def extractext(self):
        if self.docformat == "pdf":
            pdf_document = fitz.open(self.docpath)
            pageNums = pdf_document.page_count
            text_list = []
            for i in range(pageNums):
                page = pdf_document[i]
                tmptexts = page.get_text().split(" \n")
                for tt in tmptexts:
                    t = tt.strip().replace("\n", "")
                    if t is not None and len(t) > 0:
                        text_list.append(t)
            return text_list
        else:
            return None