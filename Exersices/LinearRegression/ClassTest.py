class TestClass:
    def __init__(self, num_features=2):
        self.num_features = num_features
        self.W = None

    def test(self):
        self.cache = ([1,2,3],[6,5,4])


    def PrintCache(self):
        print(self.cache[1])