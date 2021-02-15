import unittest
import puzzle

class TestPuzzle(unittest.TestCase):
    def clearFile(self, filePath):
        open(filePath, 'w').close()

    def check(self, testPath):
        expected_outputs = actual_outputs = None
        searchMode = initialState = None
        contents = None

        with open(testPath, 'r') as fin:
            contents = [ line for line in fin.readlines() ]
        print()
        searchMode, initialState = contents[0].split(';')
        print(f'Inputs: {searchMode};{initialState}')

        expected_outputs = contents[1:]
        #print(f'Expected: {expected_outputs}')

        puzzle.findPath(searchMode, initialState.split(","))

        with open('output.txt', 'r') as fin:
            actual_outputs = [ line for line in fin.readlines() ]
        
        #print(f'Actual: {actual_outputs}')
        
        for expected, actual in zip(expected_outputs, actual_outputs):
            if expected.startswith('path_to_goal') and '...' in expected:
                self.assertTrue(actual.startswith('path_to_goal'))
                expected_path = expected.split(',')
                actual_path = actual.split(',')
                index_of_elipsis = expected_path.index(' ... ')
                for expected_step, actual_step in zip(expected_path[:index_of_elipsis], actual_path[:index_of_elipsis]):
                    print(f'Comparing {expected_step} with {actual_step}')
                    self.assertEqual(expected_step, actual_step)
                for expected_step, actual_step in zip(expected_path[:index_of_elipsis:-1], actual_path[:index_of_elipsis:-1]):
                    print(f'Comparing {expected_step} with {actual_step}')
                    self.assertEqual(expected_step, actual_step)
            elif expected.startswith('running_time'):
                self.assertTrue(actual.startswith('running_time'))
            elif expected.startswith('max_ram_usage'):
                self.assertTrue(actual.startswith('max_ram_usage'))
            else:
                print(f'Comparing {expected} with {actual}')
                self.assertEqual(expected, actual)
            
    def test_bfs_a(self):
        self.clearFile('output.txt')
        self.check('testResources/bfs_a.txt')

    def test_dfs_a(self):
        self.clearFile('output.txt')
        self.check('testResources/dfs_a.txt')
            
    def test_ast_a(self):
        self.clearFile('output.txt')
        self.check('testResources/ast_a.txt')
    
    def test_bfs_b(self):
        self.clearFile('output.txt')
        self.check('testResources/bfs_b.txt')

    def test_dfs_b(self):
        self.clearFile('output.txt')
        self.check('testResources/dfs_b.txt')

    def test_ast_b(self):
        self.clearFile('output.txt')
        self.check('testResources/ast_b.txt')

    def test_bfs_c(self):
        self.clearFile('output.txt')
        self.check('testResources/bfs_c.txt')

    def test_dfs_c(self):
        self.clearFile('output.txt')
        self.check('testResources/dfs_c.txt')

    def test_ast_c(self):
        self.clearFile('output.txt')
        self.check('testResources/ast_c.txt')

    def test_bfs_d(self):
        self.clearFile('output.txt')
        self.check('testResources/bfs_d.txt')

    def test_dfs_d(self):
        self.clearFile('output.txt')
        self.check('testResources/dfs_d.txt')

    def test_ast_d(self):
        self.clearFile('output.txt')
        self.check('testResources/ast_d.txt')

    def test_bfs_e(self):
        self.clearFile('output.txt')
        self.check('testResources/bfs_e.txt')

    def test_dfs_e(self):
        self.clearFile('output.txt')
        self.check('testResources/dfs_e.txt')

    def test_ast_e(self):
        self.clearFile('output.txt')
        self.check('testResources/ast_e.txt')

if __name__ == '__main__':
    unittest.main()