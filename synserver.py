import json,base64,traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs, unquote, urlencode

from build import buildfunc

class MyHandler(BaseHTTPRequestHandler):
    #def do_POST(self):
    def do_GET(self):
        # 解析URL中的查询字符串
        print("parsing...")
        print(urlparse(self.path).query)
        query = parse_qs(urlparse(self.path).query)
        print(query)
        # 获取参数值
        taskjson = unquote(query.get('taskjson', [''])[0])
        docjson = unquote(query.get('docjson', [''])[0])
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        try:
            builtinfo = buildfunc(docjson, taskjson)
            #rrr = json.dumps(builtinfo).encode("utf-8")
            #self.wfile.write(rrr)
            self.wfile.write(bytes(builtinfo, "utf-8"))
        except Exception as e:
            self.wfile.write(bytes("<html><head><title>Python HTTP Server</title></head>", "utf-8"))
            self.wfile.write(bytes("<body><p>Exception: %s</p>" % str(e), "utf-8"))
            print(e)
            import sys
            print(traceback.print_exc(file=sys.stdout))
        
if __name__ == '__main__':
    # 启动HTTP服务器
    server_address = ('', 9433)
    httpd = HTTPServer(server_address, MyHandler)
    print('服务已开启...')
    httpd.serve_forever()