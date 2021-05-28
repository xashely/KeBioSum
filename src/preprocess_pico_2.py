from copy import deepcopy
import argparse
import os
import itertools
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-data_file", type=str)

args = parser.parse_args()

def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ''
    if empty_line:
        return True
    else:
        first_token = line.split()[0]
        if first_token == "-DOCSTART-":  # pylint: disable=simplifiable-if-statement
            return line
        else:
            return False


file_path = os.path.abspath(args.data_file)
with open(file_path, "r") as data_file:

    # Group into alternative divider / sentence chunks.
    # for idx,line in enumerate(tqdm(data_file)):
    #     if _is_divider(line):
    #         sep_line = line
    #     if not _is_divider(line):
    #         line_split = line.split(' ')
    #         # if '<bioterrorism' in line_split[0]:
    #         #     print(sep_line.strip(' ').strip('\n'), idx, line_split)
    #         if len(line_split) > 4:
    #             print(sep_line.strip(' ').strip('\n'), idx, line_split)
    fields = None
    for is_divider, lines in itertools.groupby(data_file, _is_divider):
        # Ignore the divider chunks, so that `lines` corresponds to the words
        # of a single sentence.
        if is_divider:
            sep_line_test = [line.strip().split() for line in lines]
            if len(sep_line_test[0]):
                sep_line = sep_line_test

        if not is_divider:
            fields = [line.strip().split() for line in lines]
            for val in fields:
                if len(val)!= 4:
                    print('\n\n\n\n\n')
                    print(val)
                    print(sep_line_test)
                    print('\n')
                    print(fields)

        # if is_divider:
        #     sep_line_test = [line.strip().split() for line in lines]
        #     if len(sep_line_test[0]):
        #         sep_line = sep_line_test
        # if not is_divider:
        #     prev_fields = fields
        #     fields = [line.strip().split() for line in lines]
        #     try:
        #         fields = [list(field) for field in zip(*fields)]
        #         tokens_, _, _, pico_tags = fields
        #     except:
        #         print('\n\n\n\n\n\n\nTOO LONG')
        #         print(fields)
        #         print(len(fields))
        #         print(sep_line)
        #         print('\n\n\n\n\n\n')
        #         fields = [
        #             val if len(val) == 4 else [" ".join(val[:-3]), val[-3], val[-2], val[-1]]
        #             for val in fields
        #         ]
        #         print(prev_fields)
        #         print(fields)
        #         fields = [list(field) for field in zip(*fields)]
        #         tokens_, _, _, pico_tags = fields
        #         print(fields)
            # unzipping trick returns tuples, but our Fields need lists
            
