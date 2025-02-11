from buildtask import VectorTableBuildTask

class MultiscaleVectorTableBuildTask(VectorTableBuildTask):
    def __init__(self, taskjson):
        VectorTableBuildTask.__init__(self, taskjson)
        self.scalelist = taskjson["scalelist"].split("|")
        try:
            self.scalelist = [float(x) for x in self.scalelist]
        except Exception as e:
            print(e)        
    def synbuildtext(self, textlist, pkprefix="", textsegwindowsize=None, textsegslidestep=None):
        ws = self.textsegwindowsize if textsegwindowsize is None else textsegwindowsize
        ss = self.textsegslidestep if textsegslidestep is None else textsegslidestep
        textseginfo = []
        scale = self.scalelist if 1.0 in self.scalelist else self.scalelist+[1.0]
        for scale in scale:
            seginfo = VectorTableBuildTask.synbuildtext(self, textlist, pkprefix+"scale"+str(scale), int(scale * ws), int(scale * ss))
            textseginfo += seginfo
        return textseginfo