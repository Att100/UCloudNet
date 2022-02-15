import math

class ProgressBar:
    def __init__(self, maxStep=150, fill="#"):
        self.maxStep = maxStep
        self.fill = fill
        self.barLength = 20
        self.barInterval = 0
        self.prInterval = 0
        self.count = 0
        self.progress = 0
        self.barlenSmaller = True
        self.genBarCfg()

    def genBarCfg(self):
        if self.maxStep >= self.barLength:
            self.barInterval = math.ceil(self.maxStep / self.barLength)
        else:
            self.barlenSmaller = False
            self.barInterval = math.floor(self.barLength / self.maxStep)
        self.prInterval = 100 / self.maxStep

    def resetBar(self):
        self.count = 0
        self.progress = 0

    def updateBar(self, step, headData={'head':10}, endData={'end_1':2.2, 'end_2':1.0}, keep=False):
        head_str = "\r"
        end_str = " "
        process = ""
        if self.barlenSmaller:
            if step != 0 and step % self.barInterval == 0:
                self.count += 1
        else:
            self.count += self.barInterval
        self.progress += self.prInterval
        for key in headData.keys():
            head_str = head_str + key + ": " + str(headData[key]) + " "
        for key in endData.keys():
            end_str = end_str + key + ": " + str(endData[key]) + " "
        if step == self.maxStep:
            process += head_str
            process += "[%3s%%]: [%-20s]" % (100.0, self.fill * self.barLength)
            process += end_str
            if not keep:
                process += "\n"
        else:
            process += head_str
            process += "[%3s%%]: [%-20s]" % (round(self.progress, 1), self.fill * self.count)
            process += end_str
        print(process, end='', flush=True)