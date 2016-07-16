import os, os.path

directory = './step_0/input'
num_of_files = len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

output_file_template = "./step_0/output/tweets_{}.json"
current_file = 0
file_per_set = 10000
current_output_file = output_file_template.format(current_file)

f = open(current_output_file, 'w')
start = current_file * file_per_set
end = start + file_per_set

for x in range(0, num_of_files):

	if x == end:
		f.close()

		current_file += 1
		current_output_file = output_file_template.format(current_file)
		f = open(current_output_file, 'w')

		start = current_file * file_per_set
		end = start + file_per_set

	filename = '{}/{}.json'.format(directory, x)
	with open(filename, 'r') as f2:
		print 'Reading {}'.format(filename)
		for line in f2:
			if len(line.strip()) < 1:
				continue
			f.write(line.strip())
			f.write('\n')
			f.flush()

f.close()