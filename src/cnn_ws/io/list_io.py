'''
Created on Nov 21, 2012

@author: lrothack
'''
import os
import io
import tqdm

class LineListIO(object):
    '''
    Helper class for reading/writing text files into lists.
    The elements of the list are the lines in the text file.
    '''
    @staticmethod
    def read_list(filepath, encoding='ascii'):        
        if not os.path.exists(filepath):
            raise ValueError('File for reading list does NOT exist: ' + filepath)
        
        linelist = []        
        if encoding == 'ascii':
            transform = lambda line: line.encode()
        else:
            transform = lambda line: line 

        with io.open(filepath, encoding=encoding) as stream:            
            for line in stream:
                line = transform(line.strip())
                if line != '':
                    linelist.append(line)                    
        return linelist

    @staticmethod
    def write_list(file_path, line_list, encoding='ascii', 
                   append=False, verbose=False):
        '''
        Writes a list into the given file object
        
        file_path: the file path that will be written to
        line_list: the list of strings that will be written
        '''                
        mode = 'w'
        if append:
            mode = 'a'
        
        with io.open(file_path, mode, encoding=encoding) as f:
            if verbose:
                line_list = tqdm.tqdm(line_list)
              
            for l in line_list:
                f.write(unicode(l) + '\n')
