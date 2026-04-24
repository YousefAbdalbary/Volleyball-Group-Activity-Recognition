class BoxInfo:
    def __init__(self, line):
        words = line.split()
        self.category = words.pop()
        words = [int(x) for x in words]
        self.player_ID = words[0]
        del words[0]

        x1, y1, x2, y2, frame_ID, lost, grouping, geneerated = words
        self.box = x1, y1, x2, y2
        self.frame_ID = frame_ID
        self.lost = lost
        self.grouping = grouping
        self.geneerated = geneerated
