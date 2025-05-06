
import sys
import os
import re
import math

directory = sys.argv[1]

entries_to_write = list()
new_line_end_of_line_pattern = re.compile('\n$')
min_time = math.inf
max_time = - math.inf

for filename in os.listdir('./' + directory):
    if 'combined' not in filename:
        file_object = open('./' + directory + '/' + filename)
        filename_segments = filename.replace('.log', '').split('-')
        timestamp_float = float(filename_segments[-1])
        for entry in file_object:
            if new_line_end_of_line_pattern.search(entry) is None:
                entry = entry + f",\"{filename_segments[-1]}\"\n"
            else:
                entry = entry + f",\"{filename_segments[-1]}\""
            entries_to_write.append(entry)
        file_object.close()
        if timestamp_float < min_time:
            min_time = timestamp_float
        if timestamp_float > max_time:
            max_time = timestamp_float

output_filename = './' + directory + '/combined-' + str(int(min_time)) + '-' + str(int(max_time)) + '.csv'
output_file = open(output_filename, 'w+')
output_file.write(f"\"Configuration\",\"Operation Executed\",\"Execution Timestamp\",\"Cores\",\"Memory\",\"Memory Python\",\"Execution Time\"\n")
for entry in entries_to_write:
    entry_segments = entry.split(',')
    entry_segments = list(re.sub('[^A-z0-9\._\-]+', '', entry_segment) for entry_segment in entry_segments)
    configs = entry_segments[0].split('-')
    configs = list(re.sub('[^0-9]+', '', config) for config in configs)
    output_string = f"\"{entry_segments[0]}\",\"{entry_segments[1]}\",\"{entry_segments[3]}\",\"{configs[3]}\",\"{configs[4]}\",\"{entry_segments[2]}\"\n"
    output_file.write(output_string)

output_file.close()

